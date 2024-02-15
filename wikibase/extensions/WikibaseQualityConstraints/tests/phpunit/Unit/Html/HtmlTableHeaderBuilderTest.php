<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Html;

use InvalidArgumentException;
use WikibaseQuality\ConstraintReport\Html\HtmlTableHeaderBuilder;

/**
 * @covers \WikibaseQuality\ConstraintReport\Html\HtmlTableHeaderBuilder
 *
 * @group WikibaseQuality
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class HtmlTableHeaderBuilderTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider constructDataProvider
	 */
	public function testConstruct( $content, $isSortable, $expectedException = null ) {
		if ( $expectedException !== null ) {
			$this->expectException( $expectedException );
		}
		$header = new HtmlTableHeaderBuilder( $content, $isSortable );

		$this->assertSame( $content, $header->getContent() );
		$this->assertSame( $isSortable, $header->getIsSortable() );
	}

	/**
	 * Test cases for testConstruct
	 */
	public static function constructDataProvider() {
		return [
			[
				'foobar',
				true,
			],
			[
				42,
				true,
				InvalidArgumentException::class,
			],
			[
				'fooar',
				42,
				InvalidArgumentException::class,
			],
		];
	}

	/**
	 * @dataProvider toHtmlDataProvider
	 */
	public function testToHtml( $content, $isSortable, $expectedHtml ) {
		$header = new HtmlTableHeaderBuilder( $content, $isSortable );
		$actualHtml = $header->toHtml();

		$this->assertSame( $expectedHtml, $actualHtml );
	}

	/**
	 * Test cases for testToHtml
	 */
	public static function toHtmlDataProvider() {
		return [
			[
				'foobar',
				true,
				'<th role="columnheader button">foobar</th>',
			],
			[
				'foobar',
				false,
				'<th role="columnheader button" class="unsortable">foobar</th>',
			],
		];
	}

}
