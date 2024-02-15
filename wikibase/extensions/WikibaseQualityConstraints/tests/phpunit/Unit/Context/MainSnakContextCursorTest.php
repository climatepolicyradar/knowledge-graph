<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Context;

use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ApiV2ContextCursor
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class MainSnakContextCursorTest extends \MediaWikiUnitTestCase {

	public function testStoreCheckResultInArray() {
		$statement1Guid = 'P1$13ea0742-0190-4d88-b7b0-baee67573818';
		$statement2Guid = 'P1$9fbfae7f-6f21-4967-8e2c-ec04ca16873d';
		$statement3Guid = 'P2$4638ca58-5128-4a1f-88a9-b379fe9f8ad9';
		$statement1SnakHash = 'c77761897897f63f151c4a1deb8bd3ad23ac51c6';
		$statement2SnakHash = 'c77761897897f63f151c4a1deb8bd3ad23ac51c6';
		$statement3SnakHash = '64baea7b9db49c64f5daf05ddf5162faa28d5e86';
		$cursor1 = new MainSnakContextCursor( 'Q1', 'P1', $statement1Guid, $statement1SnakHash );
		$cursor2 = new MainSnakContextCursor( 'Q1', 'P1', $statement2Guid, $statement2SnakHash );
		$cursor3 = new MainSnakContextCursor( 'Q1', 'P2', $statement3Guid, $statement3SnakHash );
		$result1 = [ 'result1' ];
		$result2 = [ 'status' => 'some status', 'result' => 'second result' ];
		$result3 = [ 3 ];
		$result4 = [ [ 'the fourth result' ] ];

		$actual = [];
		$cursor1->storeCheckResultInArray( $result1, $actual );
		$cursor2->storeCheckResultInArray( $result2, $actual );
		$cursor3->storeCheckResultInArray( $result3, $actual );
		$cursor3->storeCheckResultInArray( $result4, $actual );

		$expected = [
			'Q1' => [
				'claims' => [
					'P1' => [
						[
							'id' => $statement1Guid,
							'mainsnak' => [
								'hash' => $statement1SnakHash,
								'results' => [
									$result1,
								],
							],
						],
						[
							'id' => $statement2Guid,
							'mainsnak' => [
								'hash' => $statement2SnakHash,
								'results' => [
									$result2,
								],
							],
						],
					],
					'P2' => [
						[
							'id' => $statement3Guid,
							'mainsnak' => [
								'hash' => $statement3SnakHash,
								'results' => [
									$result3,
									$result4,
								],
							],
						],
					],
				],
			],
		];
		$this->assertSame( $expected, $actual );
	}

	public function testStoreCheckResultInArray_NullResult() {
		$statement1Guid = 'P1$13ea0742-0190-4d88-b7b0-baee67573818';
		$statement2Guid = 'P1$9fbfae7f-6f21-4967-8e2c-ec04ca16873d';
		$statement3Guid = 'P2$4638ca58-5128-4a1f-88a9-b379fe9f8ad9';
		$statement1SnakHash = 'c77761897897f63f151c4a1deb8bd3ad23ac51c6';
		$statement2SnakHash = 'c77761897897f63f151c4a1deb8bd3ad23ac51c6';
		$statement3SnakHash = '64baea7b9db49c64f5daf05ddf5162faa28d5e86';
		$cursor1 = new MainSnakContextCursor( 'Q1', 'P1', $statement1Guid, $statement1SnakHash );
		$cursor2 = new MainSnakContextCursor( 'Q1', 'P1', $statement2Guid, $statement2SnakHash );
		$cursor3 = new MainSnakContextCursor( 'Q1', 'P2', $statement3Guid, $statement3SnakHash );

		$actual = [];
		$cursor1->storeCheckResultInArray( null, $actual );
		$cursor2->storeCheckResultInArray( null, $actual );
		$cursor3->storeCheckResultInArray( null, $actual );

		$expected = [
			'Q1' => [
				'claims' => [
					'P1' => [
						[
							'id' => $statement1Guid,
							'mainsnak' => [
								'hash' => $statement1SnakHash,
								'results' => [],
							],
						],
						[
							'id' => $statement2Guid,
							'mainsnak' => [
								'hash' => $statement2SnakHash,
								'results' => [],
							],
						],
					],
					'P2' => [
						[
							'id' => $statement3Guid,
							'mainsnak' => [
								'hash' => $statement3SnakHash,
								'results' => [],
							],
						],
					],
				],
			],
		];
		$this->assertSame( $expected, $actual );
	}

	/**
	 * only getter that isn’t covered by testStoreCheckResultInArray
	 */
	public function testGetSnakPropertyId() {
		$statementGuid = 'P1$13ea0742-0190-4d88-b7b0-baee67573818';
		$statementSnakHash = 'c77761897897f63f151c4a1deb8bd3ad23ac51c6';
		$cursor = new MainSnakContextCursor( 'Q1', 'P1', $statementGuid, $statementSnakHash );

		$this->assertSame( $cursor->getStatementPropertyId(), $cursor->getSnakPropertyId() );
	}

}
