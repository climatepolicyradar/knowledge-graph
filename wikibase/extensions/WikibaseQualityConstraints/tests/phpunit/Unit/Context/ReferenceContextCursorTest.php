<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Context;

use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ApiV2ContextCursor
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContextCursor
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ReferenceContextCursorTest extends \MediaWikiUnitTestCase {

	public function testStoreCheckResultInArray() {
		$statement1Guid = 'P1$13ea0742-0190-4d88-b7b0-baee67573818';
		$statement2Guid = 'P1$9fbfae7f-6f21-4967-8e2c-ec04ca16873d';
		$statement3Guid = 'P2$4638ca58-5128-4a1f-88a9-b379fe9f8ad9';
		$snak1Hash = 'aa3ccc57c3325cba2f74f26a145f0349685945e1';
		$snak2Hash = 'de02a46e1e0afe9551f667a72f59101650a2f80e';
		$snak3Hash = '7d7f59dfb34bbc674f0f59208aa504c2930799be';
		$reference1Hash = 'c6e3c7a92ba799e27610b56532c5b7eb0a282e82';
		$reference2Hash = 'dd76d0bf8d7f9c83917866598d9a77749a0404d0';
		$cursor1 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak1Hash, 'P11', $reference1Hash );
		$cursor2 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak2Hash, 'P11', $reference1Hash );
		$cursor3 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak3Hash, 'P12', $reference1Hash );
		$cursor4 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak2Hash, 'P11', $reference2Hash );
		$cursor5 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak3Hash, 'P12', $reference2Hash );
		$cursor6 = new ReferenceContextCursor( 'Q1', 'P1', $statement2Guid, $snak1Hash, 'P11', $reference1Hash );
		$cursor7 = new ReferenceContextCursor( 'Q1', 'P2', $statement3Guid, $snak3Hash, 'P12', $reference2Hash );
		$result1 = [ 'result1' ];
		$result2 = [ 'status' => 'some status', 'result' => 'second result' ];
		$result3 = [ 3 ];
		$result4 = [ [ 'the fourth result' ] ];
		$result5 = [ [ [ 5.0 ] ] ];

		$actual = [];
		$cursor1->storeCheckResultInArray( $result1, $actual );
		$cursor2->storeCheckResultInArray( $result2, $actual );
		$cursor3->storeCheckResultInArray( $result3, $actual );
		$cursor3->storeCheckResultInArray( $result4, $actual );
		$cursor4->storeCheckResultInArray( $result4, $actual );
		$cursor5->storeCheckResultInArray( $result5, $actual );
		$cursor6->storeCheckResultInArray( $result5, $actual );
		$cursor7->storeCheckResultInArray( $result5, $actual );

		$expected = [
			'P1' => [
				[
					'id' => $statement1Guid,
					'references' => [
						[
							'hash' => $reference1Hash,
							'snaks' => [
								'P11' => [
									[
										'hash' => $snak1Hash,
										'results' => [
											$result1,
										],
									],
									[
										'hash' => $snak2Hash,
										'results' => [
											$result2,
										],
									],
								],
								'P12' => [
									[
										'hash' => $snak3Hash,
										'results' => [
											$result3,
											$result4,
										],
									],
								],
							],
						],
						[
							'hash' => $reference2Hash,
							'snaks' => [
								'P11' => [
									[
										'hash' => $snak2Hash,
										'results' => [
											$result4,
										],
									],
								],
								'P12' => [
									[
										'hash' => $snak3Hash,
										'results' => [
											$result5,
										],
									],
								],
							],
						],
					],
				],
				[
					'id' => $statement2Guid,
					'references' => [
						[
							'hash' => $reference1Hash,
							'snaks' => [
								'P11' => [
									[
										'hash' => $snak1Hash,
										'results' => [
											$result5,
										],
									],
								],
							],
						],
					],
				],
			],
			'P2' => [
				[
					'id' => $statement3Guid,
					'references' => [
						[
							'hash' => $reference2Hash,
							'snaks' => [
								'P12' => [
									[
										'hash' => $snak3Hash,
										'results' => [
											$result5,
										],
									],
								],
							],
						],
					],
				],
			],
		];
		$this->assertSame( [ 'Q1' ], array_keys( $actual ) );
		$this->assertSame( [ 'claims' ], array_keys( $actual['Q1'] ) );
		$this->assertSame( $expected, $actual['Q1']['claims'] );
	}

	public function testStoreCheckResultInArray_NullResult() {
		$statement1Guid = 'P1$13ea0742-0190-4d88-b7b0-baee67573818';
		$statement2Guid = 'P1$9fbfae7f-6f21-4967-8e2c-ec04ca16873d';
		$statement3Guid = 'P2$4638ca58-5128-4a1f-88a9-b379fe9f8ad9';
		$snak1Hash = 'aa3ccc57c3325cba2f74f26a145f0349685945e1';
		$snak2Hash = 'de02a46e1e0afe9551f667a72f59101650a2f80e';
		$snak3Hash = '7d7f59dfb34bbc674f0f59208aa504c2930799be';
		$reference1Hash = 'c6e3c7a92ba799e27610b56532c5b7eb0a282e82';
		$reference2Hash = 'dd76d0bf8d7f9c83917866598d9a77749a0404d0';
		$cursor1 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak1Hash, 'P11', $reference1Hash );
		$cursor2 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak2Hash, 'P11', $reference1Hash );
		$cursor3 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak3Hash, 'P12', $reference1Hash );
		$cursor4 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak2Hash, 'P11', $reference2Hash );
		$cursor5 = new ReferenceContextCursor( 'Q1', 'P1', $statement1Guid, $snak3Hash, 'P12', $reference2Hash );
		$cursor6 = new ReferenceContextCursor( 'Q1', 'P1', $statement2Guid, $snak1Hash, 'P11', $reference1Hash );
		$cursor7 = new ReferenceContextCursor( 'Q1', 'P2', $statement3Guid, $snak3Hash, 'P12', $reference2Hash );

		$actual = [];
		$cursor1->storeCheckResultInArray( null, $actual );
		$cursor2->storeCheckResultInArray( null, $actual );
		$cursor3->storeCheckResultInArray( null, $actual );
		$cursor3->storeCheckResultInArray( null, $actual );
		$cursor4->storeCheckResultInArray( null, $actual );
		$cursor5->storeCheckResultInArray( null, $actual );
		$cursor6->storeCheckResultInArray( null, $actual );
		$cursor7->storeCheckResultInArray( null, $actual );

		$expected = [
			'P1' => [
				[
					'id' => $statement1Guid,
					'references' => [
						[
							'hash' => $reference1Hash,
							'snaks' => [
								'P11' => [
									[
										'hash' => $snak1Hash,
										'results' => [],
									],
									[
										'hash' => $snak2Hash,
										'results' => [],
									],
								],
								'P12' => [
									[
										'hash' => $snak3Hash,
										'results' => [],
									],
								],
							],
						],
						[
							'hash' => $reference2Hash,
							'snaks' => [
								'P11' => [
									[
										'hash' => $snak2Hash,
										'results' => [],
									],
								],
								'P12' => [
									[
										'hash' => $snak3Hash,
										'results' => [],
									],
								],
							],
						],
					],
				],
				[
					'id' => $statement2Guid,
					'references' => [
						[
							'hash' => $reference1Hash,
							'snaks' => [
								'P11' => [
									[
										'hash' => $snak1Hash,
										'results' => [],
									],
								],
							],
						],
					],
				],
			],
			'P2' => [
				[
					'id' => $statement3Guid,
					'references' => [
						[
							'hash' => $reference2Hash,
							'snaks' => [
								'P12' => [
									[
										'hash' => $snak3Hash,
										'results' => [],
									],
								],
							],
						],
					],
				],
			],
		];
		$this->assertSame( [ 'Q1' ], array_keys( $actual ) );
		$this->assertSame( [ 'claims' ], array_keys( $actual['Q1'] ) );
		$this->assertSame( $expected, $actual['Q1']['claims'] );
	}

}
