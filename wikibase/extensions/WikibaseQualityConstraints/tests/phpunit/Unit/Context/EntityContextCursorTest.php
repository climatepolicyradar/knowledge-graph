<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Context;

use LogicException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\EntityContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ApiV2ContextCursor
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\EntityContextCursor
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class EntityContextCursorTest extends \MediaWikiUnitTestCase {

	public function testStoreCheckResultInArray() {
		$entityId = 'Q1';
		$cursor = new EntityContextCursor( $entityId );
		$result = [ 'result' ];

		$actual = [];
		$this->expectException( LogicException::class );
		$cursor->storeCheckResultInArray( $result, $actual );
	}

	public function testStoreCheckResultInArray_NullResult() {
		$entityId1 = 'Q1';
		$entityId2 = 'Q2';
		$cursor1 = new EntityContextCursor( $entityId1 );
		$cursor2 = new EntityContextCursor( $entityId1 );
		$cursor3 = new EntityContextCursor( $entityId2 );

		$actual = [];
		$cursor1->storeCheckResultInArray( null, $actual );
		$cursor2->storeCheckResultInArray( null, $actual );
		$cursor3->storeCheckResultInArray( null, $actual );

		$expected = [
			'Q1' => [
				'claims' => [],
			],
			'Q2' => [
				'claims' => [],
			],
		];
		$this->assertSame( $expected, $actual );
	}

}
