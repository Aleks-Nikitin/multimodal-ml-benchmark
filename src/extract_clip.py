import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.data_utils import load_cifar_subset, CIFAR10_CLASSES


def to_tensor_features(raw_out, projection=None):
    tensor = raw_out.pooler_output if hasattr(raw_out, "pooler_output") else raw_out
    if projection is not None and tensor.shape[-1] != projection.out_features:
        tensor = projection(tensor)
    return tensor.detach().cpu().numpy()


def main():
    os.makedirs("./embeddings", exist_ok=True)
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    X_train, y_train, X_test, y_test = load_cifar_subset()

    def get_image_vectors(images):
        vectors = []
        for img in images:
            inputs = processor(images=Image.fromarray(img), return_tensors="pt")
            raw_feats = model.get_image_features(**inputs)
            feats = to_tensor_features(raw_feats, getattr(model, "visual_projection", None))[0]
            feats = feats / np.linalg.norm(feats) 
            vectors.append(feats)
        return np.array(vectors)

    train_vecs = get_image_vectors(X_train)
    test_vecs = get_image_vectors(X_test)

    prompts = [f"a photo of a {name}" for name in CIFAR10_CLASSES]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt")
    raw_text = model.get_text_features(**text_inputs)
    text_vecs = to_tensor_features(raw_text, getattr(model, "text_projection", None))
    text_vecs = text_vecs / np.linalg.norm(text_vecs, axis=1, keepdims=True)

    np.save("./embeddings/train_clip.npy", train_vecs)
    np.save("./embeddings/y_train.npy", y_train)
    np.save("./embeddings/test_clip.npy", test_vecs)
    np.save("./embeddings/y_test.npy", y_test)
    np.save("./embeddings/text_clip.npy", text_vecs)
    print("Done! Saved embeddings to ./embeddings/")


if __name__ == "__main__":
    main()