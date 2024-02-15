<?php

namespace WikibaseQuality\ConstraintReport\Tests\Result;

use DataValues\Deserializers\DataValueDeserializer;
use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\TimeValue;
use PHPUnit\Framework\TestCase;
use Wikibase\DataModel\Entity\BasicEntityIdParser;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\DataValueFactory;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;
use WikibaseQuality\ConstraintReport\ConstraintDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintSerializer;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CheckResultSerializationTest extends TestCase {

	/**
	 * @dataProvider provideCheckResultsWithSerializations
	 */
	public function testSerialize( CheckResult $checkResult, array $serialization ) {
		$serializer = new CheckResultSerializer(
			new ConstraintSerializer(),
			new ContextCursorSerializer(),
			new ViolationMessageSerializer()
		);

		$serialized = $serializer->serialize( $checkResult );

		$this->assertSame( $serialization, $serialized );
	}

	/**
	 * @dataProvider provideCheckResultsWithSerializations
	 */
	public function testSerialize_withoutDependencyMetadata( CheckResult $checkResult, array $serialization ) {
		$serializer = new CheckResultSerializer(
			new ConstraintSerializer(),
			new ContextCursorSerializer(),
			new ViolationMessageSerializer(),
			false
		);

		$serialized = $serializer->serialize( $checkResult );

		unset( $serialization[CheckResultSerializer::KEY_DEPENDENCY_METADATA] );
		$this->assertSame( $serialization, $serialized );
	}

	/**
	 * @dataProvider provideCheckResultsWithSerializations
	 */
	public function testDeserialize( CheckResult $checkResult, array $serialization ) {
		$entityIdParser = new BasicEntityIdParser();
		$dataValueFactory = new DataValueFactory( new DataValueDeserializer() );
		$deserializer = new CheckResultDeserializer(
			new ConstraintDeserializer(),
			new ContextCursorDeserializer(),
			new ViolationMessageDeserializer( $entityIdParser, $dataValueFactory ),
			$entityIdParser
		);

		$deserialized = $deserializer->deserialize( $serialization );

		$this->assertEquals( $checkResult, $deserialized );
	}

	/**
	 * @dataProvider provideCheckResultsWithSerializations
	 */
	public function testDeserialize_withoutDependencyMetadata( CheckResult $checkResult, array $serialization ) {
		$entityIdParser = new BasicEntityIdParser();
		$dataValueFactory = new DataValueFactory( new DataValueDeserializer() );
		$deserializer = new CheckResultDeserializer(
			new ConstraintDeserializer(),
			new ContextCursorDeserializer(),
			new ViolationMessageDeserializer( $entityIdParser, $dataValueFactory ),
			$entityIdParser
		);
		unset( $serialization[CheckResultSerializer::KEY_DEPENDENCY_METADATA] );

		$deserialized = $deserializer->deserialize( $serialization );

		if ( $checkResult instanceof NullResult ) {
			$expected = new NullResult( $checkResult->getContextCursor() );
		} else {
			$expected = ( new CheckResult(
				$checkResult->getContextCursor(),
				$checkResult->getConstraint(),
				$checkResult->getStatus(),
				$checkResult->getMessage()
			) )->withMetadata(
				Metadata::ofCachingMetadata( $checkResult->getMetadata()->getCachingMetadata() )
			);
			$expected->setConstraintClarification( $checkResult->getConstraintClarification() );
		}
		$this->assertEquals( $expected, $deserialized );
	}

	public static function provideCheckResultsWithSerializations() {
		$contextCursor = new MainSnakContextCursor(
			'Q42',
			'P31',
			'Q42$F078E5B3-F9A8-480E-B7AC-D97778CBBEF9',
			'ad7d38a03cdd40cdc373de0dc4e7b7fcbccb31d9'
		);
		$constraint = new Constraint(
			'P370$f3ef0b09-4a58-3e69-1db6-b6abb5449f89',
			new NumericPropertyId( 'P370' ),
			'Q5',
			[]
		);

		$checkResult = new CheckResult( $contextCursor, $constraint );
		$checkResult->setConstraintClarification( new MultilingualTextValue( [
			new MonolingualTextValue( 'de', 'de clarification' ),
			new MonolingualTextValue( 'en', 'en clarification' ),
		] ) );
		yield 'unimplemented constraint type with constraint clarification' => [
			$checkResult,
			[
				CheckResultSerializer::KEY_CONTEXT_CURSOR => [
					't' => Context::TYPE_STATEMENT,
					'i' => $contextCursor->getEntityId(),
					'p' => $contextCursor->getSnakPropertyId(),
					'g' => $contextCursor->getStatementGuid(),
					'h' => $contextCursor->getSnakHash(),
				],
				CheckResultSerializer::KEY_CONSTRAINT => [
					'id' => $constraint->getConstraintId(),
					'pid' => $constraint->getPropertyId()->getSerialization(),
					'qid' => $constraint->getConstraintTypeItemId(),
					'params' => [],
				],
				CheckResultSerializer::KEY_CHECK_RESULT_STATUS => CheckResult::STATUS_TODO,
				CheckResultSerializer::KEY_CACHING_METADATA => [],
				CheckResultSerializer::KEY_CONSTRAINT_CLARIFICATION => [
					[ 'text' => 'de clarification', 'language' => 'de' ],
					[ 'text' => 'en clarification', 'language' => 'en' ],
				],
				CheckResultSerializer::KEY_DEPENDENCY_METADATA => [
					CheckResultSerializer::KEY_DEPENDENCY_METADATA_ENTITY_IDS => [],
				],
			],
		];

		$futureTime = new TimeValue(
			'+2018-04-04T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
		$constraint = new Constraint(
			'P370$1b1678e7-4871-0172-037a-98dfa9bb6986',
			new NumericPropertyId( 'P370' ),
			'Q21502404',
			[]
		);
		yield 'constraint with missing parameters' => [
			( new CheckResult(
				$contextCursor,
				$constraint,
				CheckResult::STATUS_BAD_PARAMETERS,
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( 'Q21502404' ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId(
						new NumericPropertyId( 'P1793' ),
						Role::CONSTRAINT_PARAMETER_PROPERTY
					)
			) )->withMetadata( Metadata::merge( [
				Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 5 * 60 ) ),
				Metadata::ofDependencyMetadata( DependencyMetadata::merge( [
					DependencyMetadata::ofEntityId( new ItemId( 'Q42' ) ),
					DependencyMetadata::ofFutureTime( $futureTime ),
				] ) ),
			] ) ),
			[
				CheckResultSerializer::KEY_CONTEXT_CURSOR => [
					't' => Context::TYPE_STATEMENT,
					'i' => $contextCursor->getEntityId(),
					'p' => $contextCursor->getSnakPropertyId(),
					'g' => $contextCursor->getStatementGuid(),
					'h' => $contextCursor->getSnakHash(),
				],
				CheckResultSerializer::KEY_CONSTRAINT => [
					'id' => $constraint->getConstraintId(),
					'pid' => $constraint->getPropertyId()->getSerialization(),
					'qid' => $constraint->getConstraintTypeItemId(),
					'params' => [],
				],
				CheckResultSerializer::KEY_CHECK_RESULT_STATUS => CheckResult::STATUS_BAD_PARAMETERS,
				CheckResultSerializer::KEY_CACHING_METADATA => [
					CheckResultSerializer::KEY_CACHING_METADATA_MAX_AGE => 5 * 60,
				],
				CheckResultSerializer::KEY_VIOLATION_MESSAGE => [
					'k' => 'parameter-needed',
					'a' => [
						[ 't' => ViolationMessage::TYPE_ENTITY_ID, 'v' => 'Q21502404', 'r' => Role::CONSTRAINT_TYPE_ITEM ],
						[ 't' => ViolationMessage::TYPE_ENTITY_ID, 'v' => 'P1793', 'r' => Role::CONSTRAINT_PARAMETER_PROPERTY ],
					],
				],
				CheckResultSerializer::KEY_DEPENDENCY_METADATA => [
					CheckResultSerializer::KEY_DEPENDENCY_METADATA_ENTITY_IDS => [ 'Q42' ],
					CheckResultSerializer::KEY_DEPENDENCY_METADATA_FUTURE_TIME => $futureTime->getArrayValue(),
				],
			],
		];

		yield 'NullResult' => [
			( new NullResult( $contextCursor ) )->withMetadata(
				Metadata::ofDependencyMetadata( DependencyMetadata::merge( [
					DependencyMetadata::ofEntityId( new ItemId( 'Q42' ) ),
					DependencyMetadata::ofFutureTime( $futureTime ),
				] ) ) ),
			[
				CheckResultSerializer::KEY_CONTEXT_CURSOR => [
					't' => Context::TYPE_STATEMENT,
					'i' => $contextCursor->getEntityId(),
					'p' => $contextCursor->getSnakPropertyId(),
					'g' => $contextCursor->getStatementGuid(),
					'h' => $contextCursor->getSnakHash(),
				],
				CheckResultSerializer::KEY_NULL_RESULT => 1,
				CheckResultSerializer::KEY_DEPENDENCY_METADATA => [
					CheckResultSerializer::KEY_DEPENDENCY_METADATA_ENTITY_IDS => [ 'Q42' ],
					CheckResultSerializer::KEY_DEPENDENCY_METADATA_FUTURE_TIME => $futureTime->getArrayValue(),
				],
			],
		];
	}

}
