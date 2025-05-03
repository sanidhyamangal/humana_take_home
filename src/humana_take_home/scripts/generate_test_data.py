from argparse import ArgumentParser
from dataclasses import dataclass

from llama_index.core.evaluation import DatasetGenerator

from humana_take_home.utils import get_default_ollama_llm
from humana_take_home.vectorizers.pdf import PDFDataLoader


def main(
    num_questions: int,
    input_dir: str | None = None,
    input_files: str | list[str] | None = None,
) -> None:
    pdf_data_loader = PDFDataLoader()

    documents = pdf_data_loader._load_data(input_dir=input_dir, input_files=input_files)

    dataset_generator = DatasetGenerator.from_documents(
        documents=documents, llm=get_default_ollama_llm(temperature=0.0)
    )

    return dataset_generator


if __name__ == "__main__":

    @dataclass
    class CommandLineArgs:
        num_questions: int
        input_dir: str | None
        input_files: str | list[str] | None

    argparser = ArgumentParser("Generate dataset for testing", allow_abbrev=False)
    argparser.add_argument(
        "--num_questions", help="Number of questions to generate per node", type=int
    )
    argparser.add_argument(
        "--input_dir", help="input directory to load files", required=False
    )
    argparser.add_argument(
        "--input_files",
        help="list of input files or input file to use for generating qas",
        required=False,
        nargs="*",
    )

    args = CommandLineArgs(**vars(argparser.parse_args()))

    qas = main(
        num_questions=args.num_questions,
        input_dir=args.input_dir,
        input_files=args.input_files,
    )
