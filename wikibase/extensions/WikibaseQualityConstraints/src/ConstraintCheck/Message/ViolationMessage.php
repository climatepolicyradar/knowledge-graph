<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Message;

use DataValues\DataValue;
use DataValues\MultilingualTextValue;
use InvalidArgumentException;
use Wikibase\DataModel\Entity\EntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;

/**
 * A violation message of a constraint check.
 *
 * A ViolationMessage object is immutable:
 * operations like {@link withEntityId} return a modified copy.
 *
 * @license GPL-2.0-or-later
 */
class ViolationMessage {

	/**
	 * @private
	 */
	public const MESSAGE_KEY_PREFIX = 'wbqc-violation-message-';

	/**
	 * Argument type for a single entity ID.
	 * Value type: {@link EntityId}
	 */
	public const TYPE_ENTITY_ID = 'e';

	/**
	 * Argument type for a list of entity IDs.
	 * Value type: {@link EntityId}[]
	 */
	public const TYPE_ENTITY_ID_LIST = 'E';

	/**
	 * Argument type for an item ID, “unknown value”, or “no value”.
	 * Value type: {@link ItemIdSnakValue}
	 */
	public const TYPE_ITEM_ID_SNAK_VALUE = 'i';

	/**
	 * Argument type for a list of item IDs, “unknown value”s, or “no value”s.
	 * Value type: {@link ItemIdSnakValue}[]
	 */
	public const TYPE_ITEM_ID_SNAK_VALUE_LIST = 'I';

	/**
	 * Argument type for a single data value.
	 * Value type: {@link DataValue}
	 */
	public const TYPE_DATA_VALUE = 'v';

	/**
	 * Argument type for a data value type, like "time" or "wikibase-entityid".
	 * (Not to be confused with a data type, like "time" or "wikibase-item".)
	 * Value type: string
	 */
	public const TYPE_DATA_VALUE_TYPE = 't';

	/**
	 * Argument type for a short fragment of inline computer code.
	 * Value type: string
	 */
	public const TYPE_INLINE_CODE = 'c';

	/**
	 * Argument type for a single constraint scope.
	 * Value type: string (one of the Context::TYPE_* constants)
	 */
	public const TYPE_CONSTRAINT_SCOPE = 's';

	/**
	 * Argument type for a list of constraint scopes.
	 * Value type: string[]
	 */
	public const TYPE_CONSTRAINT_SCOPE_LIST = 'S';

	/**
	 * Argument type for a single property scope.
	 * Value type: string (one of the Context::TYPE_* constants)
	 */
	public const TYPE_PROPERTY_SCOPE = 'p';

	/**
	 * Argument type for a list of property scopes.
	 * Value type: string[]
	 */
	public const TYPE_PROPERTY_SCOPE_LIST = 'P';

	/**
	 * Argument type for a language.
	 * Value type: string (language code)
	 */
	public const TYPE_LANGUAGE = 'l';

	/**
	 * Argument type for list of languages.
	 * Value type: string[] (language codes)
	 */
	public const TYPE_LANGUAGE_LIST = 'L';

	/**
	 * Argument type for a multilingual text value.
	 * Value type: {@link MultilingualTextValue}
	 */
	public const TYPE_MULTILINGUAL_TEXT = 'm';

	/**
	 * @var string
	 */
	private $messageKeySuffix;

	/**
	 * @var array[]
	 */
	private $arguments;

	/**
	 * @param string $messageKey The full message key. Must start with 'wbqc-violation-message-'.
	 * (We require callers to specify the full message key
	 * so that it’s easy to search for code that produces a given message.)
	 *
	 * @throws InvalidArgumentException If the message key is invalid.
	 */
	public function __construct(
		$messageKey
	) {
		if ( strpos( $messageKey, self::MESSAGE_KEY_PREFIX ) !== 0 ) {
			throw new InvalidArgumentException(
				'ViolationMessage key ⧼' .
				$messageKey .
				'⧽ should start with "' .
				self::MESSAGE_KEY_PREFIX .
				'".'
			);
		}

		$this->messageKeySuffix = substr( $messageKey, strlen( self::MESSAGE_KEY_PREFIX ) );
		$this->arguments = [];
	}

	/**
	 * Get the full message key of this message.
	 * @return string
	 */
	public function getMessageKey() {
		return self::MESSAGE_KEY_PREFIX . $this->messageKeySuffix;
	}

	/**
	 * Get the arguments to this message,
	 * a list of [ 'type' => self::TYPE_*, 'role' => Role::*, 'value' => $value ] elements.
	 * @return array[]
	 */
	public function getArguments() {
		return $this->arguments;
	}

	/**
	 * Note: usually you don’t want to use this function directly.
	 *
	 * @param string $type one of the self::TYPE_* constants
	 * @param string|null $role one of the Role::* constants
	 * @param mixed $value the value, which should match the $type
	 * @return ViolationMessage
	 */
	public function withArgument( $type, $role, $value ) {
		$ret = clone $this;
		$ret->arguments[] = [ 'type' => $type, 'role' => $role, 'value' => $value ];
		return $ret;
	}

	/**
	 * Append a single entity ID to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param EntityId $entityId
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withEntityId( EntityId $entityId, $role = null ) {
		return $this->withArgument( self::TYPE_ENTITY_ID, $role, $entityId );
	}

	/**
	 * Append a list of entity IDs to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * This is not the same as appending the list elements individually with {@link withEntityId}!
	 * In the final message, this corresponds to
	 * one parameter for the number of list elements,
	 * one parameter with an HTML list of the list elements,
	 * and then one parameter per entity ID.
	 *
	 * @param EntityId[] $entityIdList
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withEntityIdList( array $entityIdList, $role = null ) {
		return $this->withArgument( self::TYPE_ENTITY_ID_LIST, $role, $entityIdList );
	}

	/**
	 * Append a single item ID, “unknown value”, or “no value” to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param ItemIdSnakValue $value
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withItemIdSnakValue( ItemIdSnakValue $value, $role = null ) {
		return $this->withArgument( self::TYPE_ITEM_ID_SNAK_VALUE, $role, $value );
	}

	/**
	 * Append a list of item IDs, “unknown value”s, or “no value”s to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * This is not the same as appending the list elements individually with {@link withItemIdSnakValue}!
	 * In the final message, this corresponds to
	 * one parameter for the number of list elements,
	 * one parameter with an HTML list of the list elements,
	 * and then one parameter per value.
	 *
	 * @param ItemIdSnakValue[] $valueList
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withItemIdSnakValueList( array $valueList, $role = null ) {
		return $this->withArgument( self::TYPE_ITEM_ID_SNAK_VALUE_LIST, $role, $valueList );
	}

	/**
	 * Append a single data value to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param DataValue $dataValue
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withDataValue( DataValue $dataValue, $role = null ) {
		return $this->withArgument( self::TYPE_DATA_VALUE, $role, $dataValue );
	}

	/**
	 * Append a single data value type, like "time" or "wikibase-entityid".
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * Data value types should not be confused with data types, like "time" or "wikibase-item".
	 * For example, "wikibase-entityid" is the data value type
	 * used by the data types "wikibase-item" and "wikibase-property".
	 *
	 * @param string $dataValueType
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withDataValueType( $dataValueType, $role = null ) {
		return $this->withArgument( self::TYPE_DATA_VALUE_TYPE, $role, $dataValueType );
	}

	/**
	 * Append a single short fragment of inline computer code to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param string $code
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withInlineCode( $code, $role = null ) {
		return $this->withArgument( self::TYPE_INLINE_CODE, $role, $code );
	}

	/**
	 * Append a single constraint scope to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param string $scope one of the Context::TYPE_* constants
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withConstraintScope( $scope, $role = null ) {
		return $this->withArgument( self::TYPE_CONSTRAINT_SCOPE, $role, $scope );
	}

	/**
	 * Append a list of constraint scopes to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param string[] $scopeList Role::* constants
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withConstraintScopeList( array $scopeList, $role = null ) {
		return $this->withArgument( self::TYPE_CONSTRAINT_SCOPE_LIST, $role, $scopeList );
	}

	/**
	 * Append a single property scope to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param string $scope one of the Context::TYPE_* constants
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withPropertyScope( $scope, $role = null ) {
		return $this->withArgument( self::TYPE_PROPERTY_SCOPE, $role, $scope );
	}

	/**
	 * Append a list of property scopes to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * @param string[] $scopeList Role::* constants
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withPropertyScopeList( array $scopeList, $role = null ) {
		return $this->withArgument( self::TYPE_PROPERTY_SCOPE_LIST, $role, $scopeList );
	}

	/**
	 * Append a single language to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * One language argument corresponds to two params in the final message,
	 * one for the language name (autonym) and one for the language code.
	 *
	 * (Language arguments do not support roles.)
	 *
	 * @param string $languageCode
	 * @return ViolationMessage
	 */
	public function withLanguage( $languageCode ) {
		return $this->withArgument( self::TYPE_LANGUAGE, null, $languageCode );
	}

	/**
	 * Append a single language to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * One language argument corresponds to two params in the final message,
	 * one for the language name (autonym) and one for the language code.
	 *
	 * (Language arguments do not support roles.)
	 *
	 * @param string[] $languageCodes
	 * @return ViolationMessage
	 */
	public function withLanguages( $languageCodes ) {
		return $this->withArgument( self::TYPE_LANGUAGE_LIST, null, $languageCodes );
	}

	/**
	 * Append a multilingual text value to the message arguments.
	 * (This operation returns a modified copy, the original object is unchanged.)
	 *
	 * Note that multilingual text arguments can only be rendered for specific message keys
	 * (see {@link MultilingualTextViolationMessageRenderer} for details),
	 * but this method does not verify whether you’re using one of those message keys.
	 *
	 * @param MultilingualTextValue $text
	 * @param string|null $role one of the Role::* constants
	 * @return ViolationMessage
	 */
	public function withMultilingualText( MultilingualTextValue $text, $role = null ) {
		return $this->withArgument( self::TYPE_MULTILINGUAL_TEXT, $role, $text );
	}

}
