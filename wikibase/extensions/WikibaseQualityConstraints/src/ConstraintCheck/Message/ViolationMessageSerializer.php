<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Message;

use DataValues\DataValue;
use DataValues\MultilingualTextValue;
use InvalidArgumentException;
use LogicException;
use Serializers\Serializer;
use Wikibase\DataModel\Entity\EntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use Wikimedia\Assert\Assert;

/**
 * A serializer for {@link ViolationMessage}s.
 *
 * @license GPL-2.0-or-later
 */
class ViolationMessageSerializer implements Serializer {

	private function abbreviateViolationMessageKey( $fullMessageKey ) {
		return substr( $fullMessageKey, strlen( ViolationMessage::MESSAGE_KEY_PREFIX ) );
	}

	/**
	 * @param ViolationMessage $object
	 * @return array
	 */
	public function serialize( $object ) {
		/** @var ViolationMessage $object */
		Assert::parameterType( ViolationMessage::class, $object, '$object' );

		$arguments = $object->getArguments();
		$serializedArguments = [];
		foreach ( $arguments as $argument ) {
			$serializedArguments[] = $this->serializeArgument( $argument );
		}

		return [
			'k' => $this->abbreviateViolationMessageKey( $object->getMessageKey() ),
			'a' => $serializedArguments,
		];
	}

	/**
	 * @param array $argument element of ViolationMessage::getArguments()
	 * @return array [ 't' => ViolationMessage::TYPE_*, 'v' => serialized value, 'r' => $role ]
	 */
	private function serializeArgument( array $argument ) {
		$methods = [
			ViolationMessage::TYPE_ENTITY_ID => 'serializeEntityId',
			ViolationMessage::TYPE_ENTITY_ID_LIST => 'serializeEntityIdList',
			ViolationMessage::TYPE_ITEM_ID_SNAK_VALUE => 'serializeItemIdSnakValue',
			ViolationMessage::TYPE_ITEM_ID_SNAK_VALUE_LIST => 'serializeItemIdSnakValueList',
			ViolationMessage::TYPE_DATA_VALUE => 'serializeDataValue',
			ViolationMessage::TYPE_DATA_VALUE_TYPE => 'serializeStringByIdentity',
			ViolationMessage::TYPE_INLINE_CODE => 'serializeStringByIdentity',
			ViolationMessage::TYPE_CONSTRAINT_SCOPE => 'serializeContextType',
			ViolationMessage::TYPE_CONSTRAINT_SCOPE_LIST => 'serializeContextTypeList',
			ViolationMessage::TYPE_PROPERTY_SCOPE => 'serializeContextType',
			ViolationMessage::TYPE_PROPERTY_SCOPE_LIST => 'serializeContextTypeList',
			ViolationMessage::TYPE_LANGUAGE => 'serializeStringByIdentity',
			ViolationMessage::TYPE_LANGUAGE_LIST => 'serializeStringListByIdentity',
			ViolationMessage::TYPE_MULTILINGUAL_TEXT => 'serializeMultilingualText',
		];

		$type = $argument['type'];
		$value = $argument['value'];
		$role = $argument['role'];

		if ( array_key_exists( $type, $methods ) ) {
			$method = $methods[$type];
			$serializedValue = $this->$method( $value );
		} else {
			throw new InvalidArgumentException(
				'Unknown ViolationMessage argument type ' . $type . '!'
			);
		}

		$serialized = [
			't' => $type,
			'v' => $serializedValue,
			'r' => $role,
		];

		return $serialized;
	}

	/**
	 * @param string $string any value that shall simply be serialized to itself
	 * @return string that same value, unchanged
	 */
	private function serializeStringByIdentity( $string ) {
		Assert::parameterType( 'string', $string, '$string' );
		return $string;
	}

	/**
	 * @param string[] $strings
	 * @return string[]
	 */
	private function serializeStringListByIdentity( $strings ) {
		Assert::parameterElementType( 'string', $strings, '$strings' );
		return $strings;
	}

	/**
	 * @param EntityId $entityId
	 * @return string entity ID serialization
	 */
	private function serializeEntityId( EntityId $entityId ) {
		return $entityId->getSerialization();
	}

	/**
	 * @param EntityId[] $entityIdList
	 * @return string[] entity ID serializations
	 */
	private function serializeEntityIdList( array $entityIdList ) {
		return array_map( [ $this, 'serializeEntityId' ], $entityIdList );
	}

	/**
	 * @param ItemIdSnakValue $value
	 * @return string entity ID serialization, '::somevalue', or '::novalue'
	 * (according to EntityId::PATTERN, entity ID serializations can never begin with two colons)
	 */
	private function serializeItemIdSnakValue( ItemIdSnakValue $value ) {
		switch ( true ) {
			case $value->isValue():
				return $this->serializeEntityId( $value->getItemId() );
			case $value->isSomeValue():
				return '::somevalue';
			case $value->isNoValue():
				return '::novalue';
			default:
				// @codeCoverageIgnoreStart
				throw new LogicException(
					'ItemIdSnakValue should guarantee that one of is{,Some,No}Value() is true'
				);
				// @codeCoverageIgnoreEnd
		}
	}

	/**
	 * @param ItemIdSnakValue[] $valueList
	 * @return string[] array of entity ID serializations, '::somevalue's or '::novalue's
	 */
	private function serializeItemIdSnakValueList( array $valueList ) {
		return array_map( [ $this, 'serializeItemIdSnakValue' ], $valueList );
	}

	/**
	 * @param DataValue $dataValue
	 * @return array the data value in array form
	 */
	private function serializeDataValue( DataValue $dataValue ) {
		return $dataValue->toArray();
	}

	/**
	 * @param string $contextType one of the Context::TYPE_* constants
	 * @return string the abbreviated context type
	 */
	private function serializeContextType( $contextType ) {
		switch ( $contextType ) {
			case Context::TYPE_STATEMENT:
				return 's';
			case Context::TYPE_QUALIFIER:
				return 'q';
			case Context::TYPE_REFERENCE:
				return 'r';
			default:
				// @codeCoverageIgnoreStart
				throw new LogicException(
					'Unknown context type ' . $contextType
				);
				// @codeCoverageIgnoreEnd
		}
	}

	/**
	 * @param string[] $contextTypeList Context::TYPE_* constants
	 * @return string[] abbreviated context types
	 */
	private function serializeContextTypeList( array $contextTypeList ) {
		return array_map( [ $this, 'serializeContextType' ], $contextTypeList );
	}

	/**
	 * @param MultilingualTextValue $text
	 * @return mixed {@see MultilingualTextValue::getArrayValue}
	 */
	private function serializeMultilingualText( MultilingualTextValue $text ) {
		return $text->getArrayValue();
	}

}
