"""Classifica uma imagem como gato ou cachorro com o modelo treinado.

Uso:
    python src/predict.py --imagem caminho/para/foto.jpg
    python src/predict.py --imagem foto.jpg --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img

IMG_SIZE = (160, 160)
CLASS_NAMES = ("cat", "dog")
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "cats_vs_dogs_mobilenetv2.keras"
)


def classify(image_path: str | Path, model_path: str | Path = DEFAULT_MODEL, threshold: float = 0.5) -> dict:
    """Retorna a classe prevista e as probabilidades para uma imagem."""
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            "Execute o notebook para treinar e salvar o modelo."
        )

    model = tf.keras.models.load_model(model_path)
    image = load_img(image_path, target_size=IMG_SIZE)
    array = preprocess_input(img_to_array(image))
    prob_dog = float(model.predict(np.expand_dims(array, axis=0), verbose=0)[0][0])

    label = CLASS_NAMES[1] if prob_dog >= threshold else CLASS_NAMES[0]
    confidence = prob_dog if label == "dog" else 1.0 - prob_dog

    return {
        "imagem": str(image_path),
        "classe": label,
        "confianca": round(confidence, 4),
        "probabilidade_cachorro": round(prob_dog, 4),
        "probabilidade_gato": round(1.0 - prob_dog, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica gatos e cachorros com Transfer Learning.")
    parser.add_argument("--imagem", required=True, help="Caminho da imagem de entrada.")
    parser.add_argument("--modelo", default=str(DEFAULT_MODEL), help="Caminho do modelo .keras treinado.")
    parser.add_argument("--limiar", type=float, default=0.5, help="Limiar de decisão (padrão: 0.5).")
    parser.add_argument("--json", action="store_true", help="Imprime o resultado em JSON.")
    args = parser.parse_args()

    result = classify(args.imagem, args.modelo, args.limiar)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Imagem: {result['imagem']}")
        print(f"Classe prevista: {result['classe']} (confiança: {result['confianca']:.1%})")
        print(f"P(cachorro) = {result['probabilidade_cachorro']:.4f} | P(gato) = {result['probabilidade_gato']:.4f}")


if __name__ == "__main__":
    main()
