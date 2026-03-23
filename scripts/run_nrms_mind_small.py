"""
NRMS on MIND-small: generate utils, train, evaluate, and produce prediction.zip
"""
import os
import sys
import pickle
import numpy as np
import zipfile
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommenders.models.newsrec.newsrec_utils import word_tokenize, prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator

DATA_PATH = "/root/autodl-tmp/mind_small"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VALID_DIR = os.path.join(DATA_PATH, "valid")
UTILS_DIR = os.path.join(DATA_PATH, "utils")

EPOCHS = 5
BATCH_SIZE = 32
SEED = 42
TITLE_SIZE = 30
HIS_SIZE = 50
WORD_EMB_DIM = 300


def build_word_dict(news_files):
    """Scan news titles to build word vocabulary."""
    word_cnt = {}
    for fpath in news_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip("\n").split("\t")
                if len(fields) < 4:
                    continue
                title = fields[3]
                for w in word_tokenize(title):
                    word_cnt[w] = word_cnt.get(w, 0) + 1
    word_dict = {}
    idx = 1
    for w, cnt in sorted(word_cnt.items(), key=lambda x: x[1], reverse=True):
        word_dict[w] = idx
        idx += 1
    return word_dict


def build_uid2index(behaviors_files):
    """Scan behaviors to build user id vocabulary."""
    uid2index = {}
    idx = 1
    for fpath in behaviors_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip("\n").split("\t")
                if len(fields) < 2:
                    continue
                uid = fields[1]
                if uid not in uid2index:
                    uid2index[uid] = idx
                    idx += 1
    return uid2index


def build_embedding_matrix(word_dict, glove_path, dim=300):
    """Build embedding matrix from GloVe file."""
    embedding_matrix = np.zeros((len(word_dict) + 1, dim), dtype=np.float32)
    glove_file = os.path.join(glove_path, f"glove.6B.{dim}d.txt")
    found = 0
    print(f"Loading GloVe from {glove_file} ...")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="GloVe"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word_dict:
                vec = np.array(parts[1:], dtype=np.float32)
                embedding_matrix[word_dict[word]] = vec
                found += 1
    print(f"Found {found}/{len(word_dict)} words in GloVe")
    return embedding_matrix


def download_glove(dest_path):
    """Download GloVe embeddings, using HF mirror if direct URL fails."""
    glove_dir = os.path.join(dest_path, "glove")
    target_file = os.path.join(glove_dir, "glove.6B.300d.txt")
    if os.path.exists(target_file):
        print("GloVe already exists, skipping download.")
        return glove_dir

    os.makedirs(glove_dir, exist_ok=True)
    zip_path = os.path.join(dest_path, "glove.6B.zip")

    if not os.path.exists(zip_path):
        urls = [
            "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip",
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
        ]
        import urllib.request
        for url in urls:
            try:
                print(f"Downloading GloVe from {url} ...")
                urllib.request.urlretrieve(url, zip_path)
                print("Download complete.")
                break
            except Exception as e:
                print(f"Failed: {e}")
                continue
        else:
            raise RuntimeError("Cannot download GloVe from any source")

    print("Extracting GloVe ...")
    import zipfile as zf
    with zf.ZipFile(zip_path, "r") as z:
        z.extractall(glove_dir)
    print("GloVe extracted.")
    return glove_dir


def write_yaml(yaml_path):
    """Write NRMS config YAML."""
    content = f"""data:
    data_format: news
    support_quick_scoring: True

model:
    model_type: nrms
    title_size: {TITLE_SIZE}
    word_emb_dim: {WORD_EMB_DIM}
    his_size: {HIS_SIZE}
    npratio: 4
    dropout: 0.2
    attention_hidden_dim: 200
    head_num: 20
    head_dim: 20
    loss: cross_entropy_loss
    optimizer: adam

train:
    learning_rate: 0.0001
    epochs: {EPOCHS}
    batch_size: {BATCH_SIZE}
    show_step: 100

info:
    metrics: ['group_auc', 'mean_mrr', 'ndcg@5;10']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"YAML written to {yaml_path}")


def generate_utils():
    """Generate all utility files needed by NRMS."""
    os.makedirs(UTILS_DIR, exist_ok=True)

    train_news = os.path.join(TRAIN_DIR, "news.tsv")
    valid_news = os.path.join(VALID_DIR, "news.tsv")
    train_behaviors = os.path.join(TRAIN_DIR, "behaviors.tsv")
    valid_behaviors = os.path.join(VALID_DIR, "behaviors.tsv")

    word_dict_file = os.path.join(UTILS_DIR, "word_dict.pkl")
    uid2index_file = os.path.join(UTILS_DIR, "uid2index.pkl")
    embedding_file = os.path.join(UTILS_DIR, "embedding.npy")
    yaml_file = os.path.join(UTILS_DIR, "nrms.yaml")

    print("=" * 50)
    print("Step 1: Building word dictionary ...")
    word_dict = build_word_dict([train_news, valid_news])
    with open(word_dict_file, "wb") as f:
        pickle.dump(word_dict, f)
    print(f"  Word dict size: {len(word_dict)}, saved to {word_dict_file}")

    print("Step 2: Building user ID index ...")
    uid2index = build_uid2index([train_behaviors, valid_behaviors])
    with open(uid2index_file, "wb") as f:
        pickle.dump(uid2index, f)
    print(f"  User count: {len(uid2index)}, saved to {uid2index_file}")

    print("Step 3: Building word embedding matrix ...")
    glove_path = download_glove(DATA_PATH)
    embedding_matrix = build_embedding_matrix(word_dict, glove_path, WORD_EMB_DIM)
    np.save(embedding_file, embedding_matrix)
    print(f"  Embedding shape: {embedding_matrix.shape}, saved to {embedding_file}")

    print("Step 4: Writing YAML config ...")
    write_yaml(yaml_file)

    print("=" * 50)
    print("All utils generated successfully!")
    return word_dict_file, uid2index_file, embedding_file, yaml_file


def train_and_evaluate():
    """Train NRMS model and run evaluation."""
    train_news = os.path.join(TRAIN_DIR, "news.tsv")
    train_behaviors = os.path.join(TRAIN_DIR, "behaviors.tsv")
    valid_news = os.path.join(VALID_DIR, "news.tsv")
    valid_behaviors = os.path.join(VALID_DIR, "behaviors.tsv")

    yaml_file = os.path.join(UTILS_DIR, "nrms.yaml")
    embedding_file = os.path.join(UTILS_DIR, "embedding.npy")
    word_dict_file = os.path.join(UTILS_DIR, "word_dict.pkl")
    uid2index_file = os.path.join(UTILS_DIR, "uid2index.pkl")

    print("\n" + "=" * 50)
    print("Preparing hyperparameters ...")
    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=embedding_file,
        wordDict_file=word_dict_file,
        userDict_file=uid2index_file,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        show_step=100,
    )
    print(hparams)

    print("\nCreating NRMS model ...")
    model = NRMSModel(hparams, MINDIterator, seed=SEED)

    print("\nEvaluating before training ...")
    res = model.run_eval(valid_news, valid_behaviors)
    print(f"Before training: {res}")

    print("\nTraining ...")
    model.fit(train_news, train_behaviors, valid_news, valid_behaviors)

    print("\nFinal evaluation ...")
    res = model.run_eval(valid_news, valid_behaviors)
    print(f"After training: {res}")

    model_dir = os.path.join(DATA_PATH, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.model.save_weights(os.path.join(model_dir, "nrms_ckpt"))
    print(f"Model saved to {model_dir}")

    return model


def generate_prediction(model):
    """Generate prediction.zip for competition submission."""
    valid_news = os.path.join(VALID_DIR, "news.tsv")
    valid_behaviors = os.path.join(VALID_DIR, "behaviors.tsv")

    print("\n" + "=" * 50)
    print("Generating predictions ...")
    group_impr_indexes, group_labels, group_preds = model.run_fast_eval(
        valid_news, valid_behaviors
    )

    pred_path = os.path.join(DATA_PATH, "prediction.txt")
    with open(pred_path, "w") as f:
        for impr_index, preds in tqdm(
            zip(group_impr_indexes, group_preds), desc="Writing predictions"
        ):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = "[" + ",".join([str(i) for i in pred_rank]) + "]"
            f.write(" ".join([str(impr_index), pred_rank]) + "\n")

    zip_path = os.path.join(DATA_PATH, "prediction.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.txt")

    print(f"Prediction saved to {zip_path}")
    return zip_path


if __name__ == "__main__":
    generate_utils()
    model = train_and_evaluate()
    generate_prediction(model)
    print("\nAll done!")
