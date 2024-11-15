import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


class WikiDataProcessor:
    def __init__(self, root_dir: str = "data/wiki_zh"):
        self.root_dir = Path(root_dir)
        self.tokenizer = None

    def check_wiki_folder(self) -> None:
        """检查数据文件夹是否存在"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"找不到必需的文件夹：{self.root_dir}")

    def combine_wiki_texts(self) -> str:
        """合并所有wiki文本"""
        combined_text = ""
        file_count = 0

        for file_path in tqdm(list(self.root_dir.rglob("*")), desc="处理文件"):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if "text" in data:
                                combined_text += data["text"] + "\n"
                        except json.JSONDecodeError:
                            continue
                file_count += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

        print(f"\n处理完成！共处理了 {file_count} 个文件")
        print(f"合并后的文本总长度: {len(combined_text)} 字符")
        return combined_text

    def split_train_val(
        self, text: str, train_ratio: float = 0.9
    ) -> Tuple[str, str]:
        """分割训练集和验证集"""
        text = text[int(0.999 * len(text)) :]  # noqa
        n = len(text)
        return (
            text[: int(n * train_ratio)],
            text[int(n * train_ratio) :],  # noqa
        )

    def init_tokenizer(self, model_name: str = "bert-base-chinese") -> None:
        """初始化分词器"""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def _tokenize_chunk(self, text_chunk: str) -> np.ndarray:
        """处理文本块的辅助函数"""
        inputs = self.tokenizer(
            text_chunk,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
        )
        return inputs["input_ids"].numpy().astype(np.uint16)

    def tokenize_and_save(
        self,
        train_data: str,
        val_data: str,
        output_dir: Union[str, Path] = None,
        num_processes: int = None,
        chunk_size: int = 1000000,
    ) -> None:
        """对数据进行多进程分词并保存"""
        if self.tokenizer is None:
            self.init_tokenizer()

        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)

        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)

        # 将文本分块
        def split_text(text: str) -> list:
            return [
                text[i : i + chunk_size]  # noqa
                for i in range(0, len(text), chunk_size)
            ]

        train_chunks = split_text(train_data)
        val_chunks = split_text(val_data)

        # 使用进程池处理分块
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            print(f"使用 {num_processes} 个进程处理数据...")
            print("处理训练数据...")
            train_results = list(
                tqdm(
                    executor.map(self._tokenize_chunk, train_chunks),
                    total=len(train_chunks),
                )
            )

            print("处理验证数据...")
            val_results = list(
                tqdm(
                    executor.map(self._tokenize_chunk, val_chunks),
                    total=len(val_chunks),
                )
            )

        # 合并结果
        train_ids = np.concatenate(train_results, axis=1)
        val_ids = np.concatenate(val_results, axis=1)

        # 保存结果
        train_ids.tofile(output_dir / "train.bin")
        val_ids.tofile(output_dir / "val.bin")


def main():
    try:
        # 初始化处理器
        processor = WikiDataProcessor()
        processor.check_wiki_folder()

        # 处理文本数据
        all_text = processor.combine_wiki_texts()
        train_data, val_data = processor.split_train_val(all_text)

        # 分词并保存
        processor.tokenize_and_save(train_data, val_data)

    except FileNotFoundError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")


if __name__ == "__main__":
    main()
