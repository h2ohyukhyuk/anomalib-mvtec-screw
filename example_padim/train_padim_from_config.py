from pathlib import Path

from anomalib.engine import Engine


def main():
    # 현재 .py 파일이 있는 폴더
    current_dir = Path(__file__).resolve().parent

    # 같은 폴더의 config 파일
    config_path = current_dir / "config_python_padim.yaml"

    # YAML에서 Engine / Model / DataModule 생성
    engine, model, datamodule = Engine.from_config(
        config_path=config_path,
    )

    # CLI의 `anomalib train --config config.yaml`에 대응
    results = engine.train(
        model=model,
        datamodule=datamodule,
    )

    print("\n=== Test Results ===")
    print(results)


if __name__ == "__main__":
    main()