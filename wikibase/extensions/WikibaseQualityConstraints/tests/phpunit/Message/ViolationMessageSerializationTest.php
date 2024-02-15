<?php

namespace WikibaseQuality\ConstraintReport\Tests\Message;

use DataValues\Deserializers\DataValueDeserializer;
use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\StringValue;
use DataValues\TimeValue;
use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\BasicEntityIdParser;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\DataValueFactory;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ViolationMessageSerializationTest extends \PHPUnit\Framework\TestCase {

	/**
	 * @var ViolationMessageSerializer
	 */
	private $serializer;

	/**
	 * @var ViolationMessageDeserializer
	 */
	private $deserializer;

	public function setUp(): void {
		parent::setUp();
		$this->serializer = new ViolationMessageSerializer();
		$this->deserializer = new ViolationMessageDeserializer(
			new BasicEntityIdParser(),
			new DataValueFactory( new DataValueDeserializer( [
				UnboundedQuantityValue::getType() => UnboundedQuantityValue::class,
				TimeValue::getType() => TimeValue::class,
				StringValue::getType() => StringValue::class,
			] ) )
		);
	}

	/**
	 * @coversNothing
	 * @dataProvider provideViolationMessages
	 */
	public function testSerializeDeserialize( ViolationMessage $message ) {

		$serialized = $this->serializer->serialize( $message );
		$deserialized = $this->deserializer->deserialize( $serialized );

		$this->assertEquals( $message, $deserialized );
	}

	public static function provideViolationMessages() {
		return [
			'no arguments' => [ new ViolationMessage( 'wbqc-violation-message-single-value' ) ],
			'entity ID' => [
				( new ViolationMessage( 'wbqc-violation-message-no-qualifiers' ) )
					->withEntityId( new NumericPropertyId( 'P1' ) ),
			],
			'entity ID list' => [
				( new ViolationMessage( 'wbqc-violation-message-unique-value' ) )
					->withEntityIdList( [
						new ItemId( 'Q1' ),
						new NumericPropertyId( 'P1' ),
					] ),
			],
			'entity ID + somevalue' => [
				( new ViolationMessage( 'wbqc-violation-message-conflicts-with-claim' ) )
					->withEntityId( new NumericPropertyId( 'P1' ) )
					->withEntityId( new NumericPropertyId( 'P2' ) )
					->withItemIdSnakValue( ItemIdSnakValue::someValue() ),
			],
			'entity ID + quantities' => [
				( new ViolationMessage( 'wbqc-violation-message-range-quantity-closed' ) )
					->withEntityId( new NumericPropertyId( 'P1' ) )
					->withDataValue( UnboundedQuantityValue::newFromNumber( -10 ) )
					->withDataValue( UnboundedQuantityValue::newFromNumber( 0 ) )
					->withDataValue( UnboundedQuantityValue::newFromNumber( 10000 ) ),
			],
			'entity ID + times' => [
				( new ViolationMessage( 'wbqc-violation-message-time-closed' ) )
					->withEntityId( new NumericPropertyId( 'P2' ) )
					->withDataValue( new TimeValue( '+19997-02-08T00:00:00Z', 0, 0, 0, 0, 'gregorian' ) )
					->withDataValue( new TimeValue( '+1001-01-01T00:00:00Z', 0, 0, 0, 0, 'gregorian' ) )
					->withDataValue( new TimeValue( '+2000-12-31T00:00:00Z', 0, 0, 0, 0, 'gregorian' ) ),
			],
			'entity ID + data value types' => [
				( new ViolationMessage( 'wbqc-violation-message-value-needed-of-types-2' ) )
					->withEntityId( new ItemId( 'Q1' ), Role::CONSTRAINT_TYPE_ITEM )
					->withDataValueType( 'string' )
					->withDataValueType( 'monolingualtext' ),
			],
			'entity ID + constraint scope + constraint scope list' => [
				( new ViolationMessage( 'wbqc-violation-message-invalid-scope' ) )
					->withConstraintScope( Context::TYPE_QUALIFIER )
					->withEntityId( new ItemId( 'Q1' ) )
					->withConstraintScopeList( [ Context::TYPE_STATEMENT ] ),
			],
			'entity ID + property scope + property scope list' => [
				( new ViolationMessage( 'wbqc-violation-message-property-scope' ) )
					->withEntityId( new NumericPropertyId( 'P1' ) )
					->withPropertyScope( Context::TYPE_STATEMENT )
					->withPropertyScopeList( [ Context::TYPE_QUALIFIER, Context::TYPE_REFERENCE ] ),
			],
			'entity ID + string value + inline code + multilingual text' => [
				( new ViolationMessage( 'wbqc-violation-message-format-clarification' ) )
					->withEntityId( new NumericPropertyId( 'P1' ) )
					->withDataValue( new StringValue( 'ftp://mirror.example/mirror.html' ) )
					->withInlineCode( 'https?://[^/]+/.*' )
					->withMultilingualText( new MultilingualTextValue( [
						new MonolingualTextValue( 'en', 'HTTP or HTTPS URL' ),
						new MonolingualTextValue( 'de', 'HTTP- oder HTTPS-URL' ),
					] ) ),
			],
		];
	}

}
