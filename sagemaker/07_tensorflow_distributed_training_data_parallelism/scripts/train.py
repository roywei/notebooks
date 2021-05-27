import argparse
import logging
import os
import sys

from datasets import load_dataset
from tqdm import tqdm
from transformers.file_utils import is_sagemaker_dp_enabled
from transformers import BertTokenizer, TFBertForMaskedLM
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import re

if os.environ.get("SDP_ENABLED") or is_sagemaker_dp_enabled():
    SDP_ENABLED = True
    os.environ["SAGEMAKER_INSTANCE_TYPE"] = "p3dn.24xlarge"
    import smdistributed.dataparallel.tensorflow as sdp
else:
    SDP_ENABLED = False


def fit(model, loss, opt, train_dataset, epochs, train_batch_size, max_steps=None):
    pbar = tqdm(train_dataset)
    for i, batch in enumerate(pbar):
        with tf.GradientTape() as tape:
            inputs = batch[0]
            targets = batch[1]
            outputs = model(inputs, targets)
            loss_value = loss(targets, outputs.logits)

        if SDP_ENABLED:
            tape = sdp.DistributedGradientTape(tape, sparse_as_dense=True)

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        pbar.set_description(f"Loss: {loss_value.numpy().sum():.4f}")

        if SDP_ENABLED:
            if i == 0:
                sdp.broadcast_variables(model.variables, root_rank=0)
                sdp.broadcast_variables(opt.variables(), root_rank=0)

        if max_steps and i >= max_steps:
            break

    train_results = {"loss": loss_value.numpy()}
    return train_results


# Bert MLM example from https://keras.io/examples/nlp/masked_language_modeling/
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2: vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer



def get_masked_input_and_labels(encoded_texts, mask_token_id):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights



def get_datasets():
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])

    vectorize_layer = get_vectorize_layer(
        train_dataset['text'],
        28996,
        512,
        special_tokens=["[MASK]"],
    )

    # Get mask token id for masked language model
    mask_token_id = vectorize_layer(["[MASK]"]).numpy()[0][0]

    # Prepare data for masked language model
    x_all_review = vectorize_layer(train_dataset["text"]).numpy()
    x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
        x_all_review, mask_token_id
    )
    mlm_ds = tf.data.Dataset.from_tensor_slices(
        (x_masked_train, y_masked_labels, sample_weights)
    )
    mlm_ds = mlm_ds.shuffle(1000).batch(args.train_batch_size)

    if SDP_ENABLED:
        tf_train_dataset = mlm_ds.shard(sdp.size(), sdp.rank())

    return tf_train_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if SDP_ENABLED:
        sdp.init()

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], "GPU")

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForMaskedLM.from_pretrained('bert-base-cased')

    # get datasets
    tf_train_dataset = get_datasets()

    # fine optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE, from_logits=True
    )
    model.compile(optimizer=optimizer, loss=loss)

    # Training
    if args.do_train:

        # train_results = model.fit(tf_train_dataset, epochs=args.epochs, batch_size=args.train_batch_size)
        train_results = fit(
            model, loss, optimizer, tf_train_dataset, args.epochs, args.train_batch_size, max_steps=None
        )
        logger.info("*** Train ***")

        output_eval_file = os.path.join(args.output_data_dir, "train_results.txt")

        if not SDP_ENABLED or sdp.rank() == 0:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Train results *****")
                logger.info(train_results)
                for key, value in train_results.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

