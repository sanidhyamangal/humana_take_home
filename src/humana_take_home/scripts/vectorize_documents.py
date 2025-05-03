from argparse import ArgumentParser
from dataclasses import dataclass

from humana_take_home.vectorizers.pdf import PDFDataLoader


def main(
    vector_path: str, input_dir: str | None, input_files: str | list[str] | None
) -> None:
    pdf_dataloader = PDFDataLoader(vector_index_path=vector_path)

    pdf_dataloader.build_vector_index(
        input_dir=input_dir, input_files=input_files, persist_index_path=vector_path
    )


if __name__ == "__main__":

    @dataclass
    class CommandLineArgs:
        input_dir: str | None
        input_files: str | list[str] | None
        vector_path: str

    argparser = ArgumentParser("vectorize documents into text embeddings")
    argparser.add_argument(
        "--input_dir", help="directory path to index all the documents", required=False
    )
    argparser.add_argument(
        "--input_files", help="list of files to index", nargs="*", required=False
    )
    argparser.add_argument("--vector_path", help="path to flush vectorized indexes")

    args = CommandLineArgs(**vars(argparser.parse_args()))

    main(
        vector_path=args.vector_path,
        input_dir=args.input_dir,
        input_files=args.input_files,
    )
