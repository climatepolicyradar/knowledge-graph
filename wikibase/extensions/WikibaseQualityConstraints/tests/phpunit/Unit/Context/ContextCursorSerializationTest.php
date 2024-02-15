<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Context;

use InvalidArgumentException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\EntityContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorSerializer
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorDeserializer
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ContextCursorSerializationTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider provideContextCursors
	 */
	public function testSerialize( ContextCursor $cursor, array $serialization ) {
		$serializer = new ContextCursorSerializer();

		$serialized = $serializer->serialize( $cursor );

		$this->assertSame( $serialization, $serialized );
	}

	/**
	 * @dataProvider provideContextCursors
	 */
	public function testDeserialize( ContextCursor $cursor, array $serialization ) {
		$deserializer = new ContextCursorDeserializer();

		$deserialized = $deserializer->deserialize( $serialization );

		$this->assertEquals( $cursor, $deserialized );
	}

	public function testDeserialize_unknownType() {
		$serialization = [ 't' => 'unknown' ];
		$deserializer = new ContextCursorDeserializer();

		$this->expectException( InvalidArgumentException::class );
		$deserializer->deserialize( $serialization );
	}

	public static function provideContextCursors() {
		$entityId = 'Q1';
		$snakHash = '85266b6fea10b59470d6e5b39b1ba52712822ba8';
		$statementPropertyId = 'P580';
		$statementGuid = 'Q1$789eef0c-4108-cdda-1a63-505cdd324564';

		yield 'statement context' => [
			new MainSnakContextCursor(
				$entityId,
				$statementPropertyId,
				$statementGuid,
				$snakHash
			),
			[
				't' => Context::TYPE_STATEMENT,
				'i' => $entityId,
				'p' => $statementPropertyId,
				'g' => $statementGuid,
				'h' => $snakHash,
			],
		];

		$snakHash = '795a0965a5fb08644610daba94b77779b68d45fa';
		$snakPropertyId = 'P459';

		yield 'qualifier context' => [
			new QualifierContextCursor(
				$entityId,
				$statementPropertyId,
				$statementGuid,
				$snakHash,
				$snakPropertyId
			),
			[
				't' => Context::TYPE_QUALIFIER,
				'i' => $entityId,
				'p' => $statementPropertyId,
				'g' => $statementGuid,
				'h' => $snakHash,
				'P' => $snakPropertyId,
			],
		];

		$snakHash = 'ffbb0f8fa052cafdfc7d10b5ca4652e378cd8ecf';
		$snakPropertyId = 'P248';
		$referenceHash = 'a9896160828b25b3d0942cf73df6c5bcd22cc6a8';

		yield 'reference context' => [
			new ReferenceContextCursor(
				$entityId,
				$statementPropertyId,
				$statementGuid,
				$snakHash,
				$snakPropertyId,
				$referenceHash
			),
			[
				't' => Context::TYPE_REFERENCE,
				'i' => $entityId,
				'p' => $statementPropertyId,
				'g' => $statementGuid,
				'h' => $snakHash,
				'P' => $snakPropertyId,
				'r' => $referenceHash,
			],
		];

		yield 'entity context' => [
			new EntityContextCursor(
				$entityId
			),
			[
				't' => '\entity',
				'i' => $entityId,
			],
		];
	}

}
