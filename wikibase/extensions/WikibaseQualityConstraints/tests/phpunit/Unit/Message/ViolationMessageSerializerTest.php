<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Message;

use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\StringValue;
use InvalidArgumentException;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Entity\SerializableEntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\Role;
use Wikimedia\TestingAccessWrapper;

/**
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ViolationMessageSerializerTest extends \MediaWikiUnitTestCase {

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serialize
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::abbreviateViolationMessageKey
	 */
	public function testSerialize_noArguments() {
		$message = new ViolationMessage( 'wbqc-violation-message-single-value' );
		$serializer = new ViolationMessageSerializer();

		$serialized = $serializer->serialize( $message );

		$this->assertSame(
			[ 'k' => 'single-value', 'a' => [] ],
			$serialized
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serialize
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeArgument
	 */
	public function testSerialize_entityId_withRole() {
		$message = ( new ViolationMessage( 'wbqc-violation-message-no-qualifiers' ) )
			->withEntityId( new NumericPropertyId( 'P1' ), Role::CONSTRAINT_PROPERTY );
		$serializer = new ViolationMessageSerializer();

		$serialized = $serializer->serialize( $message );

		$this->assertSame(
			[ 'k' => 'no-qualifiers', 'a' => [ [
				't' => ViolationMessage::TYPE_ENTITY_ID,
				'v' => 'P1',
				'r' => Role::CONSTRAINT_PROPERTY,
			] ] ],
			$serialized
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serialize
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeArgument
	 */
	public function testSerialize_unknownArgument() {
		$message = $this->createMock( ViolationMessage::class );
		$message->method( 'getMessageKey' )
			->willReturn( 'wbqc-violation-message-unknown-argument-type' );
		$message->method( 'getArguments' )
			->willReturn( [ [ 'type' => 'unknown', 'value' => null, 'role' => null ] ] );
		$serializer = new ViolationMessageSerializer();

		$this->expectException( InvalidArgumentException::class );
		$serializer->serialize( $message );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeStringByIdentity
	 */
	public function testSerializeStringByIdentity_noEscaping() {
		$value = '<pseudo html>&apos;; DROP TABLE Students; -- <![CDATA[ \write18{reboot} ]]>';
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeStringByIdentity( $value );

		$this->assertSame( $value, $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeStringByIdentity
	 */
	public function testSerializeStringByIdentity_empty() {
		$value = '';
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeStringByIdentity( $value );

		$this->assertSame( $value, $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeEntityId
	 */
	public function testSerializeEntityId() {
		$entityId = new NumericPropertyId( 'P1' );
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeEntityId( $entityId );

		$this->assertSame( 'P1', $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeEntityIdList
	 */
	public function testSerializeEntityIdList() {
		$entityIds = [ new ItemId( 'Q1' ), new NumericPropertyId( 'P1' ) ];
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeEntityIdList( $entityIds );

		$this->assertSame( [ 'Q1', 'P1' ], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeEntityIdList
	 */
	public function testSerializeEntityIdList_empty() {
		$entityIds = [];
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeEntityIdList( $entityIds );

		$this->assertSame( [], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeItemIdSnakValue
	 */
	public function testSerializeItemIdSnakValue_itemId() {
		$value = ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) );
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeItemIdSnakValue( $value );

		$this->assertSame( 'Q1', $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeItemIdSnakValue
	 */
	public function testSerializeItemIdSnakValue_someValue() {
		$value = ItemIdSnakValue::someValue();
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeItemIdSnakValue( $value );

		$this->assertSame( '::somevalue', $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeItemIdSnakValue
	 */
	public function testSerializeItemIdSnakValue_noValue() {
		$value = ItemIdSnakValue::noValue();
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeItemIdSnakValue( $value );

		$this->assertSame( '::novalue', $serialized );
	}

	/**
	 * Verify that a string beginning with two colons is not a valid entity ID serialization.
	 * If this ever changes, we’re in trouble because our serialization of
	 * ItemIdSnakValue::someValue() and ItemIdSnakValue::noValue() might become ambiguous.
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeItemIdSnakValue
	 */
	public function testSerializeItemIdSnakValue_senseCheck() {
		$this->expectException( InvalidArgumentException::class );

		// Since EntityId is an interface, not all EntityId implementations are guaranteed to use this constructor
		$this->getMockBuilder( SerializableEntityId::class )
			->setConstructorArgs( [ '::somevalue' ] )
			->getMockForAbstractClass();
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeItemIdSnakValueList
	 */
	public function testSerializeItemIdSnakValueList() {
		$valueList = [
			ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ),
			ItemIdSnakValue::someValue(),
			ItemIdSnakValue::noValue(),
		];
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeItemIdSnakValueList( $valueList );

		$this->assertSame( [ 'Q1', '::somevalue', '::novalue' ], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeItemIdSnakValueList
	 */
	public function testSerializeItemIdSnakValueList_empty() {
		$valueList = [];
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeItemIdSnakValueList( $valueList );

		$this->assertSame( [], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeDataValue
	 */
	public function testSerializeDataValue() {
		$dataValue = new StringValue( '<a string>' );
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeDataValue( $dataValue );

		$this->assertSame( [ 'value' => '<a string>', 'type' => 'string' ], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeContextType
	 * @dataProvider provideContextTypes
	 */
	public function testSerializeContextType( $contextType, $abbreviation ) {
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeContextType( $contextType );

		$this->assertSame( $abbreviation, $serialized );
	}

	public static function provideContextTypes() {
		return [
			[ Context::TYPE_STATEMENT, 's' ],
			[ Context::TYPE_QUALIFIER, 'q' ],
			[ Context::TYPE_REFERENCE, 'r' ],
		];
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeContextTypeList
	 */
	public function testSerializeContextTypeList() {
		$contextTypeList = [
			Context::TYPE_STATEMENT,
			Context::TYPE_REFERENCE,
			Context::TYPE_QUALIFIER,
		];
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeContextTypeList( $contextTypeList );

		$this->assertSame( [ 's', 'r', 'q' ], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeContextTypeList
	 */
	public function testSerializeContextTypeList_empty() {
		$contextTypeList = [];
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeContextTypeList( $contextTypeList );

		$this->assertSame( [], $serialized );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeMultilingualText
	 */
	public function testSerializeMultilingualText() {
		$text = new MultilingualTextValue( [
			new MonolingualTextValue( 'en', 'the text' ),
			new MonolingualTextValue( 'de', 'der Text' ),
			new MonolingualTextValue( 'html', '<span class="text"></span>' ),
		] );
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeMultilingualText( $text );

		$this->assertSame(
			[
				[ 'text' => 'the text', 'language' => 'en' ],
				[ 'text' => 'der Text', 'language' => 'de' ],
				[ 'text' => '<span class="text"></span>', 'language' => 'html' ],
			],
			$serialized
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer::serializeMultilingualText
	 */
	public function testSerializeMultilingualText_empty() {
		$text = new MultilingualTextValue( [] );
		$serializer = new ViolationMessageSerializer();

		$serialized = TestingAccessWrapper::newFromObject( $serializer )
			->serializeMultilingualText( $text );

		$this->assertSame( [], $serialized );
	}

}
