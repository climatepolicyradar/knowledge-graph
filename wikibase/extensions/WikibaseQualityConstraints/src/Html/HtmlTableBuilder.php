<?php

namespace WikibaseQuality\ConstraintReport\Html;

use Html;
use InvalidArgumentException;
use Wikimedia\Assert\Assert;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class HtmlTableBuilder {

	/**
	 * @var HtmlTableHeaderBuilder[]
	 */
	private $headers = [];

	/**
	 * Array of HtmlTableCellBuilder arrays
	 *
	 * @var array[]
	 */
	private $rows = [];

	/**
	 * @var bool
	 */
	private $isSortable;

	/**
	 * @param array $headers
	 */
	public function __construct( array $headers ) {
		foreach ( $headers as $header ) {
			$this->addHeader( $header );
		}
	}

	/**
	 * @param string|HtmlTableHeaderBuilder $header
	 *
	 * @throws InvalidArgumentException
	 */
	private function addHeader( $header ) {
		Assert::parameterType( [ 'string', HtmlTableHeaderBuilder::class ], $header, '$header' );

		if ( is_string( $header ) ) {
			$header = new HtmlTableHeaderBuilder( $header );
		}

		$this->headers[] = $header;

		if ( $header->getIsSortable() ) {
			$this->isSortable = true;
		}
	}

	/**
	 * @return HtmlTableHeaderBuilder[]
	 */
	public function getHeaders() {
		return $this->headers;
	}

	/**
	 * @return array[]
	 */
	public function getRows() {
		return $this->rows;
	}

	/**
	 * @return bool
	 */
	public function isSortable() {
		return $this->isSortable;
	}

	/**
	 * Adds row with specified cells to table.
	 *
	 * @param string[]|HtmlTableCellBuilder[] $cells
	 *
	 * @throws InvalidArgumentException
	 */
	public function appendRow( array $cells ) {
		foreach ( $cells as $key => $cell ) {
			if ( is_string( $cell ) ) {
				$cells[$key] = new HtmlTableCellBuilder( $cell );
			} elseif ( !( $cell instanceof HtmlTableCellBuilder ) ) {
				throw new InvalidArgumentException( '$cells must be array of HtmlTableCell objects.' );
			}
		}

		$this->rows[] = $cells;
	}

	/**
	 * Adds rows with specified cells to table.
	 *
	 * @param array[] $rows
	 *
	 * @throws InvalidArgumentException
	 */
	public function appendRows( array $rows ) {
		foreach ( $rows as $cells ) {
			if ( !is_array( $cells ) ) {
				throw new InvalidArgumentException( '$rows must be array of arrays of HtmlTableCell objects.' );
			}

			$this->appendRow( $cells );
		}
	}

	/**
	 * Returns table as html.
	 *
	 * @return string
	 */
	public function toHtml() {
		// Open table
		$tableClasses = 'wikitable';
		if ( $this->isSortable ) {
			$tableClasses .= ' sortable';
		}
		$html = Html::openElement( 'table', [ 'class' => $tableClasses ] );

		// Write headers
		$html .= Html::openElement( 'thead' );
		$html .= Html::openElement( 'tr' );
		foreach ( $this->headers as $header ) {
			$html .= $header->toHtml();
		}
		$html .= Html::closeElement( 'tr' );
		$html .= Html::closeElement( 'thead' );
		$html .= Html::openElement( 'tbody' );

		// Write rows
		foreach ( $this->rows as $row ) {
			$html .= Html::openElement( 'tr' );

			/**
			 * @var HtmlTableCellBuilder $cell
			 */
			foreach ( $row as $cell ) {
				$html .= $cell->toHtml();
			}

			$html .= Html::closeElement( 'tr' );
		}

		// Close table
		$html .= Html::closeElement( 'tbody' );
		$html .= Html::closeElement( 'table' );

		return $html;
	}

}
