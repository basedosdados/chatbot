import xml.etree.ElementTree as ET
from typing import Protocol

from google.cloud.bigquery.dataset import Dataset
from google.cloud.bigquery.table import Table


class MetadataFormatter(Protocol):
    @staticmethod
    def format_dataset_metadata(dataset: Dataset, tables: list[Table]) -> str:
        ...

    @staticmethod
    def format_table_metadata(table: Table, sample_rows: list[dict]) -> str:
        ...

class MarkdownMetadataFormatter:
    @staticmethod
    def format_dataset_metadata(dataset: Dataset, tables: list[Table]) -> str:
        """Format a Google BigQuery dataset's metadata in Markdown

        Args:
            dataset (Dataset): The dataset
            tables (list[Table]): A list of tables from the dataset

        Returns:
            str: Dataset metadata formatted in Markdown
        """
        # Dataset name and description
        metadata = f"# {dataset.dataset_id}\n\n### Description:\n{dataset.description}\n\n### Tables:\n"

        # Dataset tables
        tables_metadata = [
            f"- {table.full_table_id.replace(":", ".")}: {table.description}" for table in tables
        ]

        metadata += "\n\n".join(tables_metadata)

        return metadata

    @staticmethod
    def format_table_metadata(table: Table, sample_rows: list[dict]) -> str:
        """Format a Google BigQuery table's metadata in Markdown

        Args:
            table (Table): The table
            table_description (str): The table description. This arg Will be deprecated soon
            sample_rows (list[dict]): A list of sample rows from the table

        Returns:
            str: Table metadata formatted in Markdown
        """
        # Table id
        metadata = f"# {table.full_table_id.replace(":", ".")}\n\n"

        # Table description
        metadata += f"### Description:\n{table.description}\n\n"

        # Table schema
        metadata += f"### Schema:\n"
        fields = "\n\t".join([
            f"{field.name} {field.field_type}"
            for field in table.schema
        ])
        metadata += f"CREATE TABLE {table.table_id} (\n\t{fields}\n)\n\n"

        # Table columns details
        metadata += f"### Column Details:\n"
        header = "|column name|column type|column description|\n|---|---|---|"
        lines = "\n".join([
            f"|{field.name}|{field.field_type}|{field.description}|"
            for field in table.schema
        ])
        metadata += f"{header}\n{lines}\n\n"

        sample_rows_md = []

        for row in sample_rows:
            values = map(str, row.values())
            values = f"|{'|'.join(values)}|"
            sample_rows_md.append(values)

        sample_rows_md = "\n".join(sample_rows_md)
        sample_header = f"|{'|'.join(row.keys())}|\n{'|---'*len(row.keys())}|"

        metadata += f"### Sample rows:\n{sample_header}\n{sample_rows_md}"

        return metadata

class XMLMetadataFormatter:
    @staticmethod
    def format_dataset_metadata(dataset: Dataset, tables: list[Table]) -> str:
        """Format a Google BigQuery dataset's metadata in XML

        Args:
            dataset (Dataset): The dataset
            tables (list[Table]): A list of tables from the dataset

        Returns:
            str: Dataset metadata formatted in XML
        """
        # Root dataset element
        dataset_elem = ET.Element("dataset")

        # Dataset name
        name_elem = ET. SubElement(dataset_elem, "name")
        name_elem.text = dataset.dataset_id

        # Dataset description
        description_elem = ET.SubElement(dataset_elem, "description")
        description_elem.text = dataset.description

        # Dataset tables
        tables_elem = ET.SubElement(dataset_elem, "tables")
        for table in tables:
            table_elem = ET.SubElement(tables_elem, "table")

            table_name = ET.SubElement(table_elem, "name")
            table_name.text = table.full_table_id.replace(":", ".")

            table_desc = ET.SubElement(table_elem, "description")
            table_desc.text = table.description

        ET.indent(dataset_elem, space="    ")

        return ET.tostring(dataset_elem, encoding="unicode")

    @staticmethod
    def format_table_metadata(table: Table, sample_rows: list[dict]) -> str:
        """Format a Google BigQuery table's metadata in XML

        Args:
            table (Table): The table
            table_description (str): The table description. This arg Will be deprecated soon
            sample_rows (list[dict]): A list of sample rows from the table

        Returns:
            str: Table metadata formatted in XML
        """
        # Root table element
        table_elem = ET.Element("table")

        # Table id
        id_elem = ET.SubElement(table_elem, "name")
        id_elem.text = table.full_table_id.replace(":", ".")

        # Table description
        description_elem = ET.SubElement(table_elem, "description")
        description_elem.text = table.description

        # Schema
        schema_elem = ET.SubElement(table_elem, "schema")
        for field in table.schema:
            column_elem = ET.SubElement(schema_elem, "column")

            col_name = ET.SubElement(column_elem, "name")
            col_name.text = field.name

            col_type = ET.SubElement(column_elem, "type")
            col_type.text = field.field_type

            col_description = ET.SubElement(column_elem, "description")
            col_description.text = field.description

        # Sample rows
        sample_rows_elem = ET.SubElement(table_elem, "sample_rows")
        for row in sample_rows:
            row_elem = ET.SubElement(sample_rows_elem, "row")
            for col_name, col_value in row.items():
                col_elem = ET.SubElement(row_elem, col_name)
                col_elem.text = str(col_value)

        ET.indent(table_elem, space="    ")

        return ET.tostring(table_elem, encoding="unicode")

class MetadataFormatterFactory:
    @staticmethod
    def get_metadata_formatter(format: str) -> MetadataFormatter:
        match format:
            case "markdown":
                return MarkdownMetadataFormatter()
            case "xml":
                return XMLMetadataFormatter()
            case _:
                raise ValueError(f"Format should be one of the following: 'markdown', 'xml'")
