from dataclasses import dataclass

import pandas as pd
from numpy import array  # Import numpy array to handle the 'array' keyword in eval
from rich import print


@dataclass
class QAData:
    question: str
    answer: str
    context: str

    @property
    def formated_question(self):
        return f"Context:\n{self.context}\nQuestion:\n{self.question}"


def format_context(context_dict):
    """Formats the context from a dictionary into a readable text format."""
    formatted_context = ""
    titles = context_dict["title"].tolist()  # Convert title array to list
    paragraphs = context_dict["sentences"]  # List of numpy arrays of sentences

    for title, sentences_array in zip(titles, paragraphs):
        formatted_context += f"Title: {title}\n"
        formatted_context += "Paragraph:\n"
        formatted_context += (
            "\n".join(sentences_array.tolist()) + "\n\n"
        )  # Convert sentences array to list

    return formatted_context


def parse_context(context_str):
    """Parses the context string using eval and converts it to a dictionary."""
    try:
        # Use eval with numpy array handling
        context = eval(context_str)
    except Exception as e:
        raise ValueError(f"Error parsing context: {e}")

    return context


def get_nth_data_from_dataframe(data: pd.DataFrame, n: int):
    """Retrieves the nth entry from the DataFrame and formats it into a dataclass."""
    # Check if n is within bounds
    if n >= len(data) or n < 0:
        raise IndexError(
            f"Index {n} is out of bounds for data with length {len(data)}."
        )

    # Get the nth entry
    entry = data.iloc[n].to_dict()

    # Extract relevant fields
    question = entry["question"]
    answer = entry.get(
        "answer", "No answer available"
    )  # Test sets may not have an answer

    # Parse the context from string to a usable format
    context_dict = parse_context(entry["context"])

    # Format the context for readability
    context = format_context(context_dict)

    # Create and return the dataclass
    return QAData(question=question, answer=answer, context=context)
