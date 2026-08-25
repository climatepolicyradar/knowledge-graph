from datetime import date
from enum import Enum
from typing import (
    Annotated,
    Any,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field


class VerticalFlipError(Exception):
    """Exception for when a vertical flip fails."""

    pass


class BlockType(str, Enum):
    """
    List of possible block types from the PubLayNet model.

    https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html#model-label-map
    """

    TEXT = "Text"
    TITLE = "Title"
    LIST = "List"
    TABLE = "Table"
    TABLE_CELL = "TableCell"
    FIGURE = "Figure"
    INFERRED = "Inferred from gaps"
    AMBIGUOUS = "Ambiguous"
    GOOGLE_BLOCK = "Google Text Block"
    PAGE_HEADER = "pageHeader"
    PAGE_FOOTER = "pageFooter"
    TITLE_LOWER_CASE = "title"
    SECTION_HEADING = "sectionHeading"
    PAGE_NUMBER = "pageNumber"
    DOCUMENT_HEADER = "Document Header"
    FOOT_NOTE = "footnote"


class _TextBlockProto(Protocol):
    """
    Protocol capturing the shared interface of text block types.

    All text blocks (TextBlock, TextBlockV2, HTMLTextBlock) share these
    attributes/methods from _TextBlockMixin.
    """

    language: Optional[str]

    def to_string(self) -> str: ...

    def model_dump_json(
        self, *, exclude: Union[set[str], dict[str, Any], None] = None
    ) -> str: ...


class _TextBlockMixin:
    """
    Shared fields for TextBlock* classes.

    Must be used with a class that inherits from Pydantic's BaseModel.
    """

    language: Optional[str] = (
        None  # TODO: validate this against a list of language ISO codes
    )
    type: BlockType
    type_confidence: float = Field(ge=0, le=1)


class TextBlock(_TextBlockMixin, BaseModel):
    """
    text block with text as a list (v1).

    :attribute text: list of text lines contained in the text block
    """

    text_block_id: str
    text: List[str]

    def to_string(self) -> str:
        """Returns lines in a text block separated by spaces as a string."""

        return " ".join([line.strip() for line in self.text])


class TextBlockV2(_TextBlockMixin, BaseModel):
    """
    text block with text as singular (v2).

    :attribute text: text lines in the text block
    """

    text: str

    def to_string(self) -> str:
        """
        Returns lines in a text block separated by spaces as a string.

        For backwards compatibility with v1.
        """

        return self.text


class HTMLTextBlock(TextBlock):
    """
    Text block parsed from an HTML document.

    Type is set to "Text" with a confidence of 1.0 by default, as we do not predict
    types for text blocks parsed from HTML.
    """


class PDFTextBlock(TextBlock):
    """
    Text block parsed from a PDF document.

    Stores the text and positional information for a single text block extracted from
    a document.

    :attribute coords: list of coordinates of the vertices defining the boundary of
    the text block. Each coordinate is a tuple in the format (x, y). (0, 0) is at the
    top left corner of the page, and the positive x- and y- directions are right and
    down. :attribute page_number: page number of the page containing the text block.
    """

    coords: List[Tuple[float, float]]
    page_number: int = Field(ge=0)

    def to_string(self) -> str:
        """Returns lines in a text block separated by spaces as a string."""

        return " ".join([line.strip() for line in self.text])


class Page(BaseModel):
    """Bounding boxes for a specific page."""

    class BoundingBox(BaseModel):
        """A bounding box defined by a specific number coordinate points."""

        class Coordinate(BaseModel):
            """A single (x, y) coordinate point."""

            x: Annotated[float, Field(ge=0, description="X dimension of point.")]
            y: Annotated[float, Field(ge=0, description="Y dimension of point.")]

        coordinates: Annotated[
            list[Coordinate],
            Field(
                min_length=4,
                max_length=4,
                description="A restricted number of coordinates to represent the bounding box.",
            ),
        ]

    number: Annotated[
        int,
        Field(ge=0, description="Page number this entry corresponds to."),
    ]

    bounding_boxes: Annotated[
        list[BoundingBox],
        Field(
            min_length=1,
            description="List of bounding boxes on this page.",
        ),
    ]


class PDFTextBlockV2(TextBlockV2):
    """V2 text block parsed from a PDF document with str text."""

    id: Annotated[
        str,
        Field(description="Global ID. Replaces `text_block_id`."),
    ]

    idx: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
            description="Index of this text block within the range of all text blocks on the parent document",
        ),
    ]

    pages: Annotated[
        list[Page],
        Field(
            min_length=1,
            description="Page(s) within the document that this text block is found on.",
        ),
    ]

    heading_id: Optional[str] = None
    tokens: Optional[list[str]] = None
    serialised_text: Optional[str] = None

    def to_string(self) -> str:
        """
        Returns the text content as a string.

        For backwards compatibility with v1.
        """
        return self.text


class HTMLData(BaseModel):
    """Set of metadata specific to HTML documents."""

    detected_title: Optional[str] = None
    detected_date: Optional[date] = None
    has_valid_text: bool
    text_blocks: Sequence[HTMLTextBlock]


class PDFPageMetadata(BaseModel):
    """
    Set of metadata for a single page of a PDF document.

    :attribute dimensions: (width, height) of the page in pixels
    """

    page_number: int = Field(ge=0)
    dimensions: Tuple[float, float]


class PDFData(BaseModel):
    """
    Set of metadata unique to PDF documents.

    :attribute pages: List of pages contained in the document :attribute filename:
    Name of the PDF file, without extension :attribute md5sum: md5sum of PDF content
    :attribute language: list of 2-letter ISO language codes, optional. If null,
    the OCR processor didn't support language detection
    """

    page_metadata: Sequence[PDFPageMetadata]
    md5sum: str
    text_blocks: Sequence[PDFTextBlock]


class PDFDataV2(BaseModel):
    """
    Set of metadata unique to PDF documents.

    :attribute pages: List of pages contained in the document :attribute filename:
    Name of the PDF file, without extension :attribute md5sum: md5sum of PDF content
    :attribute language: list of 2-letter ISO language codes, optional. If null,
    the OCR processor didn't support language detection
    """

    page_metadata: Sequence[PDFPageMetadata]
    md5sum: str
    text_blocks: Sequence[PDFTextBlockV2]


PDFDataT = TypeVar("PDFDataT", PDFData, PDFDataV2)
PDFTextBlockT = TypeVar("PDFTextBlockT", PDFTextBlock, PDFTextBlockV2)


class _BaseParserOutputFieldsMixin(BaseModel, Generic[PDFDataT, PDFTextBlockT]):
    """Shared fields and methods for BaseParserOutput* classes."""

    document_id: str
    document_content_type: Optional[str] = None
    languages: Optional[Sequence[str]] = None
    html_data: Optional[HTMLData] = None
    pdf_data: Optional[PDFDataT] = None

    @property
    def text_blocks(self) -> Sequence[_TextBlockProto]:
        """
        Return the text blocks in the document.

        These could differ in format depending on the content type.
        """

        if self.html_data is not None:
            return self.html_data.text_blocks
        elif self.pdf_data is not None:
            return self.pdf_data.text_blocks
        return []

    def get_text_blocks(self) -> Sequence[_TextBlockProto]:
        """A method for getting text blocks."""
        return self.text_blocks

    def to_string(self) -> str:
        """Return the text blocks in the parser output as a string"""

        return " ".join(
            [text_block.to_string().strip() for text_block in self.text_blocks]
        )


class BaseParserOutput(_BaseParserOutputFieldsMixin[PDFData, PDFTextBlock]):
    """Base class for an output to a parser (v1)."""


class BaseParserOutputV2(_BaseParserOutputFieldsMixin[PDFDataV2, PDFTextBlockV2]):
    """Base class for an output from a parser (v2)."""
