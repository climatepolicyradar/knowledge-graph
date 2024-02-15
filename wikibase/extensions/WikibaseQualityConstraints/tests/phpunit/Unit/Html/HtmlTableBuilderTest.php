<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Html;

use InvalidArgumentException;
use WikibaseQuality\ConstraintReport\Html\HtmlTableBuilder;
use WikibaseQuality\ConstraintReport\Html\HtmlTableCellBuilder;
use WikibaseQuality\ConstraintReport\Html\HtmlTableHeaderBuilder;

/**
 * @covers \WikibaseQuality\ConstraintReport\Html\HtmlTableBuilder
 *
 * @group WikibaseQuality
 *
 * @uses   \WikibaseQuality\ConstraintReport\Html\HtmlTableHeaderBuilder
 * @uses   \WikibaseQuality\ConstraintReport\Html\HtmlTableCellBuilder
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class HtmlTableBuilderTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider constructDataProvider
	 */
	public function testConstruct(
		$headers,
		?array $expectedHeaders,
		$expectedIsSortable,
		$expectedException
	) {
		if ( $expectedException !== null ) {
			$this->expectException( $expectedException );
		}
		$htmlTable = new HtmlTableBuilder( $headers );

		$this->assertEquals( $expectedHeaders, $htmlTable->getHeaders() );
		$this->assertEquals( $expectedIsSortable, $htmlTable->isSortable() );
	}

	public static function constructDataProvider() {
		return [
			[
				[
					'foo',
					'bar',
				],
				[
					new HtmlTableHeaderBuilder( 'foo' ),
					new HtmlTableHeaderBuilder( 'bar' ),
				],
				false,
				null,
			],
			[
				[
					new HtmlTableHeaderBuilder( 'foo', true ),
					'bar',
				],
				[
					new HtmlTableHeaderBuilder( 'foo', true ),
					new HtmlTableHeaderBuilder( 'bar' ),
				],
				true,
				null,
			],
			[
				[
					new HtmlTableHeaderBuilder( 'foo', true ),
					new HtmlTableHeaderBuilder( 'bar' ),
				],
				[
					new HtmlTableHeaderBuilder( 'foo', true ),
					new HtmlTableHeaderBuilder( 'bar' ),
				],
				true,
				null,
			],
			[
				[ 42 ],
				null,
				false,
				InvalidArgumentException::class,
			],
		];
	}

	public function testAppendRow() {
		$htmlTable = new HtmlTableBuilder( [ 'fu', 'bar' ] );
		$htmlTable->appendRow( [ 'foo', 'bar' ] );

		$this->assertEquals(
			[
				[
					new HtmlTableCellBuilder( 'foo' ),
					new HtmlTableCellBuilder( 'bar' ),
				],
			],
			$htmlTable->getRows()
		);
	}

	/**
	 * @dataProvider appendRowsDataProvider
	 */
	public function testAppendRows(
		array $rows,
		array $expectedRows = null,
		$expectedException = null
	) {
		if ( $expectedException ) {
			$this->expectException( $expectedException );
		}

		$htmlTable = new HtmlTableBuilder( [ 'fu', 'bar' ] );
		$htmlTable->appendRows( $rows );

		$this->assertEquals( $expectedRows, $htmlTable->getRows() );
	}

	/**
	 * Test cases for testAppendRows
	 */
	public static function appendRowsDataProvider() {
		return [
			[
				[
					[
						'foo',
						'bar',
					],
				],
				[
					[
						new HtmlTableCellBuilder( 'foo' ),
						new HtmlTableCellBuilder( 'bar' ),
					],
				],
			],
			[
				[
					[
						new HtmlTableCellBuilder( 'foo' ),
						'bar',
					],
				],
				[
					[
						new HtmlTableCellBuilder( 'foo' ),
						new HtmlTableCellBuilder( 'bar' ),
					],
				],
			],
			[
				[
					[
						'foo',
						42,
					],
				],
				null,
				InvalidArgumentException::class,
			],
			[
				[
					42,
				],
				null,
				InvalidArgumentException::class,
			],
		];
	}

	/**
	 * @dataProvider toHtmlDataProvider
	 */
	public function testToHtml( $headers, $rows, $expectedHtml ) {
		// Create table
		$htmlTable = new HtmlTableBuilder( $headers );
		$htmlTable->appendRows( $rows );

		// Run assertions
		$actualHtml = $htmlTable->toHtml();
		$this->assertSame( $expectedHtml, $actualHtml );
	}

	public function toHtmlDataProvider() {
		return [
			[
				[
					$this->getHtmlTableHeaderMock( 'fu' ),
					$this->getHtmlTableHeaderMock( 'bar' ),
				],
				[
					[
						$this->getHtmlTableCellMock( 'fucked up' ),
						$this->getHtmlTableCellMock( 'beyond all recognition' ),
					],
				],
				'<table class="wikitable">'
					. '<thead><tr><th>fu</th><th>bar</th></tr></thead>'
					. '<tbody><tr><td>fucked up</td><td>beyond all recognition</td></tr></tbody>'
					. '</table>',
			],
			[
				[
					$this->getHtmlTableHeaderMock( 'fu' ),
					$this->getHtmlTableHeaderMock( 'bar', true ),
				],
				[
					[
						$this->getHtmlTableCellMock( 'fucked up' ),
						$this->getHtmlTableCellMock( 'beyond all recognition' ),
					],
				],
				'<table class="wikitable sortable">'
					. '<thead><tr><th>fu</th><th>bar</th></tr></thead>'
					. '<tbody><tr><td>fucked up</td><td>beyond all recognition</td></tr></tbody>'
					. '</table>',
			],
		];
	}

	/**
	 * Creates HtmlHeaderCell mock, which returns only the content when calling HtmlHeaderCell::toHtml()
	 *
	 * @param string $content
	 * @param bool $isSortable
	 *
	 * @return HtmlTableHeaderBuilder
	 */
	private function getHtmlTableHeaderMock( $content, $isSortable = false ) {
		$cellMock = $this
			->getMockBuilder( HtmlTableHeaderBuilder::class )
			->setConstructorArgs( [ $content, $isSortable ] )
			->onlyMethods( [ 'toHtml' ] )
			->getMock();
		$cellMock->method( 'toHtml' )
			->willReturn( "<th>$content</th>" );

		return $cellMock;
	}

	/**
	 * Creates HtmlTableCell mock, which returns only the content when calling HtmlTableCell::toHtml()
	 *
	 * @param string $content
	 *
	 * @return HtmlTableCellBuilder
	 */
	private function getHtmlTableCellMock( $content ) {
		$cellMock = $this
			->getMockBuilder( HtmlTableCellBuilder::class )
			->setConstructorArgs( [ $content ] )
			->getMock();
		$cellMock->method( 'toHtml' )
			->willReturn( "<td>$content</td>" );

		return $cellMock;
	}

}
