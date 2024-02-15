<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Html;

use InvalidArgumentException;
use WikibaseQuality\ConstraintReport\Html\HtmlTableCellBuilder;

/**
 * @covers \WikibaseQuality\ConstraintReport\Html\HtmlTableCellBuilder
 *
 * @group WikibaseQuality
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class HtmlTableCellBuilderTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider constructDataProvider
	 */
	public function testConstruct( $content, $attributes, $expectedException = null ) {
		if ( $expectedException !== null ) {
			$this->expectException( $expectedException );
		}
		$cell = new HtmlTableCellBuilder( $content, $attributes );

		$this->assertEquals( $content, $cell->getContent() );
		$this->assertEquals( $attributes, $cell->getAttributes() );
	}

	/**
	 * Test cases for testConstruct
	 */
	public static function constructDataProvider() {
		return [
			[
				'foobar',
				[],
			],
			[
				'foobar',
				[
					'rowspan' => 2,
					'colspan' => 2,
				],
			],
			[
				42,
				[],
				InvalidArgumentException::class,
			],
		];
	}

	/**
	 * @dataProvider toHtmlDataProvider
	 */
	public function testToHtml( $content, $attributes, $expectedHtml ) {
		$cell = new HtmlTableCellBuilder( $content, $attributes );
		$actualHtml = $cell->toHtml();

		$this->assertSame( $expectedHtml, $actualHtml );
	}

	/**
	 * Test cases for testToHtml
	 */
	public static function toHtmlDataProvider() {
		return [
			[
				'foobar',
				[],
				'<td>foobar</td>',
			],
			[
				'foobar',
				[
					'rowspan' => 2,
					'colspan' => 3,
				],
				'<td rowspan="2" colspan="3">foobar</td>',
			],
			[
				'foobar',
				[
					'foo' => 'bar',
				],
				'<td foo="bar">foobar</td>',
			],
		];
	}

}
