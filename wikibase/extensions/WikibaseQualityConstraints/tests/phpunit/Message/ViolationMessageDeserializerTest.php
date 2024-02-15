<?php

namespace WikibaseQuality\ConstraintReport\Tests\Message;

use DataValues\Deserializers\DataValueDeserializer;
use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\StringValue;
use InvalidArgumentException;
use Wikibase\DataModel\Entity\BasicEntityIdParser;
use Wikibase\DataModel\Entity\EntityIdParser;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\DataValueFactory;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer;
use WikibaseQuality\ConstraintReport\Role;
use Wikimedia\TestingAccessWrapper;

/**
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ViolationMessageDeserializerTest extends \PHPUnit\Framework\TestCase {

	private function getViolationMessageDeserializer(
		EntityIdParser $entityIdParser = null,
		DataValueFactory $dataValueFactory = null
	) {
		if ( $entityIdParser === null ) {
			$entityIdParser = new BasicEntityIdParser();
		}
		if ( $dataValueFactory === null ) {
			$dataValueFactory = new DataValueFactory( new DataValueDeserializer() );
		}

		return new ViolationMessageDeserializer( $entityIdParser, $dataValueFactory );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserialize
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::unabbreviateViolationMessageKey
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::__construct
	 */
	public function testDeserialize_noArguments() {
		$serialized = [ 'k' => 'single-value', 'a' => [] ];
		$deserializer = $this->getViolationMessageDeserializer();

		$message = $deserializer->deserialize( $serialized );

		$this->assertSame(
			'wbqc-violation-message-single-value',
			$message->getMessageKey()
		);
		$this->assertSame( [], $message->getArguments() );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserialize
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeArgument
	 */
	public function testDeserialize_entityId() {
		$serialized = [ 'k' => 'no-qualifiers', 'a' => [ [
			't' => ViolationMessage::TYPE_ENTITY_ID,
			'v' => 'P1',
			'r' => Role::CONSTRAINT_PROPERTY,
		] ] ];
		$deserializer = $this->getViolationMessageDeserializer();

		$message = $deserializer->deserialize( $serialized );
		$this->assertEquals(
			[ [
				'type' => ViolationMessage::TYPE_ENTITY_ID,
				'role' => Role::CONSTRAINT_PROPERTY,
				'value' => new NumericPropertyId( 'P1' ),
			] ],
			$message->getArguments()
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserialize
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeArgument
	 */
	public function testDeserialize_unknownArgument() {
		$serialized = [
			'k' => 'unknown-argument-type',
			'a' => [ [
				't' => 'unknown',
				'v' => null,
				'r' => null,
			] ],
		];
		$deserializer = $this->getViolationMessageDeserializer();

		$this->expectException( InvalidArgumentException::class );
		$deserializer->deserialize( $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeStringByIdentity
	 */
	public function testDeserializeStringByIdentity_noEscaping() {
		$serialized = '<pseudo html>&apos;; DROP TABLE Students; -- <![CDATA[ \write18{reboot} ]]>';
		$serializer = $this->getViolationMessageDeserializer();

		$value = TestingAccessWrapper::newFromObject( $serializer )
			->deserializeStringByIdentity( $serialized );

		$this->assertSame( $serialized, $value );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeStringByIdentity
	 */
	public function testDeserializeStringByIdentity_empty() {
		$serialized = '';
		$serializer = $this->getViolationMessageDeserializer();

		$value = TestingAccessWrapper::newFromObject( $serializer )
			->deserializeStringByIdentity( $serialized );

		$this->assertSame( $serialized, $value );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeEntityId
	 */
	public function testDeserializeEntityId() {
		$propertyId = new NumericPropertyId( 'P1' );
		$mock = $this->createMock( EntityIdParser::class );
		$mock->expects( $this->once() )
			->method( 'parse' )
			->with( 'P1' )
			->willReturn( $propertyId );
		$deserializer = $this->getViolationMessageDeserializer( $mock );

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeEntityId( 'P1' );

		$this->assertSame( $propertyId, $deserialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeEntityIdList
	 */
	public function testDeserializeEntityIdList() {
		$entityIdSerializations = [ 'Q1', 'P1' ];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeEntityIdList( $entityIdSerializations );

		$this->assertEquals(
			[ new ItemId( 'Q1' ), new NumericPropertyId( 'P1' ) ],
			$deserialized
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeEntityIdList
	 */
	public function testDeserializeEntityIdList_Empty() {
		$entityIdSerializations = [];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeEntityIdList( $entityIdSerializations );

		$this->assertEquals( [], $deserialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeItemIdSnakValue
	 */
	public function testDeserializeItemIdSnakValue_itemId() {
		$serialization = 'Q1';
		$deserializer = $this->getViolationMessageDeserializer();

		/** @var ItemIdSnakValue $deserialized */
		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeItemIdSnakValue( $serialization );

		$this->assertTrue( $deserialized->isValue() );
		$this->assertEquals( new ItemId( 'Q1' ), $deserialized->getItemId() );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeItemIdSnakValue
	 */
	public function testDeserializeItemIdSnakValue_someValue() {
		$serialization = '::somevalue';
		$deserializer = $this->getViolationMessageDeserializer();

		/** @var ItemIdSnakValue $deserialized */
		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeItemIdSnakValue( $serialization );

		$this->assertTrue( $deserialized->isSomeValue() );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeItemIdSnakValue
	 */
	public function testDeserializeItemIdSnakValue_noValue() {
		$serialization = '::novalue';
		$deserializer = $this->getViolationMessageDeserializer();

		/** @var ItemIdSnakValue $deserialized */
		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeItemIdSnakValue( $serialization );

		$this->assertTrue( $deserialized->isNoValue() );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeItemIdSnakValueList
	 */
	public function testDeserializeItemIdSnakValueList() {
		$serializations = [ 'Q1', '::somevalue', '::novalue' ];
		$deserializer = $this->getViolationMessageDeserializer();

		/** @var ItemIdSnakValue[] $deserialized */
		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeItemIdSnakValueList( $serializations );

		$this->assertCount( 3, $deserialized );
		$this->assertTrue( $deserialized[0]->isValue() );
		$this->assertEquals( new ItemId( 'Q1' ), $deserialized[0]->getItemId() );
		$this->assertTrue( $deserialized[1]->isSomeValue() );
		$this->assertTrue( $deserialized[2]->isNoValue() );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeItemIdSnakValueList
	 */
	public function testDeserializeItemIdSnakValueList_empty() {
		$serializations = [];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeItemIdSnakValueList( $serializations );

		$this->assertSame( [], $deserialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeDataValue
	 */
	public function testDeserializeDataValue() {
		$serialization = [ 'type' => 'string', 'value' => '<a string>' ];
		$deserializer = $this->getViolationMessageDeserializer(
			null,
			new DataValueFactory( new DataValueDeserializer( [ 'string' => StringValue::class ] ) )
		);

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeDataValue( $serialization );

		$this->assertEquals( new StringValue( '<a string>' ), $deserialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeContextType
	 * @dataProvider provideContextTypeAbbreviations
	 */
	public function testDeserializeContextType( $abbreviation, $contextType ) {
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeContextType( $abbreviation );

		$this->assertSame( $contextType, $deserialized );
	}

	public static function provideContextTypeAbbreviations() {
		return [
			[ 's', Context::TYPE_STATEMENT ],
			[ 'q', Context::TYPE_QUALIFIER ],
			[ 'r', Context::TYPE_REFERENCE ],
		];
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeContextTypeList
	 */
	public function testDeserializeContextTypeList() {
		$serializations = [ 's', 'r', 'q' ];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeContextTypeList( $serializations );

		$expected = [
			Context::TYPE_STATEMENT,
			Context::TYPE_REFERENCE,
			Context::TYPE_QUALIFIER,
		];
		$this->assertSame( $expected, $deserialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeContextTypeList
	 */
	public function testDeserializeContextTypeList_empty() {
		$serializations = [];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeContextTypeList( $serializations );

		$this->assertSame( [], $deserialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeMultilingualText
	 */
	public function testDeserializeMultilingualText() {
		$textSerialization = [
			[ 'language' => 'en', 'text' => 'must <strong>not</strong> contain punctuation' ],
			[ 'language' => 'de', 'text' => 'darf <strong>keine</strong> Satzzeichen enthalten' ],
		];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeMultilingualText( $textSerialization );

		$this->assertEquals(
			new MultilingualTextValue( [
				new MonolingualTextValue( 'en', 'must <strong>not</strong> contain punctuation' ),
				new MonolingualTextValue( 'de', 'darf <strong>keine</strong> Satzzeichen enthalten' ),
			] ),
			$deserialized
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer::deserializeMultilingualText
	 */
	public function testDeserializeMultilingualText_empty() {
		$textSerialization = [];
		$deserializer = $this->getViolationMessageDeserializer();

		$deserialized = TestingAccessWrapper::newFromObject( $deserializer )
			->deserializeMultilingualText( $textSerialization );

		$this->assertEquals( new MultilingualTextValue( [] ), $deserialized );
	}

}
