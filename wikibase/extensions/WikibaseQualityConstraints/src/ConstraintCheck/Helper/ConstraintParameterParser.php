<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Config;
use DataValues\DataValue;
use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\StringValue;
use DataValues\UnboundedQuantityValue;
use LogicException;
use Wikibase\DataModel\Deserializers\DeserializerFactory;
use Wikibase\DataModel\Deserializers\SnakDeserializer;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Entity\PropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\Snak;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\Role;

/**
 * Helper for parsing constraint parameters
 * that were imported from constraint statements.
 *
 * All public methods of this class expect constraint parameters
 * (see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()})
 * and return parameter objects or throw {@link ConstraintParameterException}s.
 * The results are used by the checkers,
 * which may include rendering them into violation messages.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ConstraintParameterParser {

	private Config $config;
	private SnakDeserializer $snakDeserializer;
	private string $unitItemConceptBaseUri;

	/**
	 * @param Config $config
	 *   contains entity IDs used in constraint parameters (constraint statement qualifiers)
	 * @param DeserializerFactory $factory
	 *   used to parse constraint statement qualifiers into constraint parameters
	 * @param string $unitItemConceptBaseUri
	 *   concept base URI of items used for units
	 */
	public function __construct(
		Config $config,
		DeserializerFactory $factory,
		string $unitItemConceptBaseUri
	) {
		$this->config = $config;
		$this->snakDeserializer = $factory->newSnakDeserializer();
		$this->unitItemConceptBaseUri = $unitItemConceptBaseUri;
	}

	/**
	 * Check if any errors are recorded in the constraint parameters.
	 * @throws ConstraintParameterException
	 */
	public function checkError( array $parameters ): void {
		if ( array_key_exists( '@error', $parameters ) ) {
			$error = $parameters['@error'];
			if ( array_key_exists( 'toolong', $error ) && $error['toolong'] ) {
				$msg = 'wbqc-violation-message-parameters-error-toolong';
			} else {
				$msg = 'wbqc-violation-message-parameters-error-unknown';
			}
			throw new ConstraintParameterException( new ViolationMessage( $msg ) );
		}
	}

	/**
	 * Require that $parameters contains exactly one $parameterId parameter.
	 * @throws ConstraintParameterException
	 */
	private function requireSingleParameter( array $parameters, string $parameterId ): void {
		if ( count( $parameters[$parameterId] ) !== 1 ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-single' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}
	}

	/**
	 * Require that $snak is a {@link PropertyValueSnak}.
	 * @throws ConstraintParameterException
	 */
	private function requireValueParameter( Snak $snak, string $parameterId ): void {
		if ( !( $snak instanceof PropertyValueSnak ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-value' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}
	}

	/**
	 * Parse a single entity ID parameter.
	 * @throws ConstraintParameterException
	 */
	private function parseEntityIdParameter( array $snakSerialization, string $parameterId ): EntityId {
		$snak = $this->snakDeserializer->deserialize( $snakSerialization );
		$this->requireValueParameter( $snak, $parameterId );
		$value = $snak->getDataValue();
		if ( $value instanceof EntityIdValue ) {
			return $value->getEntityId();
		} else {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-entity' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withDataValue( $value, Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return string[] class entity ID serializations
	 */
	public function parseClassParameter( array $constraintParameters, string $constraintTypeItemId ): array {
		$this->checkError( $constraintParameters );
		$classId = $this->config->get( 'WBQualityConstraintsClassId' );
		if ( !array_key_exists( $classId, $constraintParameters ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId( new NumericPropertyId( $classId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		$classes = [];
		foreach ( $constraintParameters[$classId] as $class ) {
			$classes[] = $this->parseEntityIdParameter( $class, $classId )->getSerialization();
		}
		return $classes;
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return string 'instance', 'subclass', or 'instanceOrSubclass'
	 */
	public function parseRelationParameter( array $constraintParameters, string $constraintTypeItemId ): string {
		$this->checkError( $constraintParameters );
		$relationId = $this->config->get( 'WBQualityConstraintsRelationId' );
		if ( !array_key_exists( $relationId, $constraintParameters ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId( new NumericPropertyId( $relationId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		$this->requireSingleParameter( $constraintParameters, $relationId );
		$relationEntityId = $this->parseEntityIdParameter( $constraintParameters[$relationId][0], $relationId );
		if ( !( $relationEntityId instanceof ItemId ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-item' ) )
					->withEntityId( new NumericPropertyId( $relationId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withDataValue( new EntityIdValue( $relationEntityId ), Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
		return $this->mapItemId( $relationEntityId, [
			$this->config->get( 'WBQualityConstraintsInstanceOfRelationId' ) => 'instance',
			$this->config->get( 'WBQualityConstraintsSubclassOfRelationId' ) => 'subclass',
			$this->config->get( 'WBQualityConstraintsInstanceOrSubclassOfRelationId' ) => 'instanceOrSubclass',
		], $relationId );
	}

	/**
	 * Parse a single property ID parameter.
	 * @param array $snakSerialization
	 * @param string $parameterId used in error messages
	 * @throws ConstraintParameterException
	 * @return PropertyId
	 */
	private function parsePropertyIdParameter( array $snakSerialization, string $parameterId ): PropertyId {
		$snak = $this->snakDeserializer->deserialize( $snakSerialization );
		$this->requireValueParameter( $snak, $parameterId );
		$value = $snak->getDataValue();
		if ( $value instanceof EntityIdValue ) {
			$id = $value->getEntityId();
			if ( $id instanceof PropertyId ) {
				return $id;
			}
		}
		throw new ConstraintParameterException(
			( new ViolationMessage( 'wbqc-violation-message-parameter-property' ) )
				->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
				->withDataValue( $value, Role::CONSTRAINT_PARAMETER_VALUE )
		);
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 *
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return PropertyId
	 */
	public function parsePropertyParameter( array $constraintParameters, string $constraintTypeItemId ): PropertyId {
		$this->checkError( $constraintParameters );
		$propertyId = $this->config->get( 'WBQualityConstraintsPropertyId' );
		if ( !array_key_exists( $propertyId, $constraintParameters ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId( new NumericPropertyId( $propertyId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		$this->requireSingleParameter( $constraintParameters, $propertyId );
		return $this->parsePropertyIdParameter( $constraintParameters[$propertyId][0], $propertyId );
	}

	private function parseItemIdParameter( PropertyValueSnak $snak, string $parameterId ): ItemIdSnakValue {
		$dataValue = $snak->getDataValue();
		if ( $dataValue instanceof EntityIdValue ) {
			$entityId = $dataValue->getEntityId();
			if ( $entityId instanceof ItemId ) {
				return ItemIdSnakValue::fromItemId( $entityId );
			}
		}
		throw new ConstraintParameterException(
			( new ViolationMessage( 'wbqc-violation-message-parameter-item' ) )
				->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
				->withDataValue( $dataValue, Role::CONSTRAINT_PARAMETER_VALUE )
		);
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @param bool $required whether the parameter is required (error if absent) or not ([] if absent)
	 * @param string|null $parameterId the property ID to use, defaults to 'qualifier of property constraint'
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return ItemIdSnakValue[] array of values
	 */
	public function parseItemsParameter(
		array $constraintParameters,
		string $constraintTypeItemId,
		bool $required,
		string $parameterId = null
	): array {
		$this->checkError( $constraintParameters );
		if ( $parameterId === null ) {
			$parameterId = $this->config->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		}
		if ( !array_key_exists( $parameterId, $constraintParameters ) ) {
			if ( $required ) {
				throw new ConstraintParameterException(
					( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
						->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
						->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
				);
			} else {
				return [];
			}
		}

		$values = [];
		foreach ( $constraintParameters[$parameterId] as $parameter ) {
			$snak = $this->snakDeserializer->deserialize( $parameter );
			switch ( true ) {
				case $snak instanceof PropertyValueSnak:
					$values[] = $this->parseItemIdParameter( $snak, $parameterId );
					break;
				case $snak instanceof PropertySomeValueSnak:
					$values[] = ItemIdSnakValue::someValue();
					break;
				case $snak instanceof PropertyNoValueSnak:
					$values[] = ItemIdSnakValue::noValue();
					break;
			}
		}
		return $values;
	}

	/**
	 * Parse a parameter that must contain item IDs.
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @param bool $required whether the parameter is required (error if absent) or not ([] if absent)
	 * @param string $parameterId the property ID to use
	 * @throws ConstraintParameterException
	 * @return ItemId[]
	 */
	private function parseItemIdsParameter(
		array $constraintParameters,
		string $constraintTypeItemId,
		bool $required,
		string $parameterId
	): array {
		return array_map( static function ( ItemIdSnakValue $value ) use ( $parameterId ): ItemId {
			if ( $value->isValue() ) {
				return $value->getItemId();
			} else {
				throw new ConstraintParameterException(
					( new ViolationMessage( 'wbqc-violation-message-parameter-value' ) )
						->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
				);
			}
		}, $this->parseItemsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			$required,
			$parameterId
		) );
	}

	/**
	 * Map an item ID parameter to a well-known value or throw an appropriate error.
	 * @throws ConstraintParameterException
	 * @return mixed elements of $mapping
	 */
	private function mapItemId( ItemId $itemId, array $mapping, string $parameterId ) {
		$serialization = $itemId->getSerialization();
		if ( array_key_exists( $serialization, $mapping ) ) {
			return $mapping[$serialization];
		} else {
			$allowed = array_map( static function ( $id ) {
				return new ItemId( $id );
			}, array_keys( $mapping ) );
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-oneof' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withEntityIdList( $allowed, Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return PropertyId[]
	 */
	public function parsePropertiesParameter( array $constraintParameters, string $constraintTypeItemId ): array {
		$this->checkError( $constraintParameters );
		$propertyId = $this->config->get( 'WBQualityConstraintsPropertyId' );
		if ( !array_key_exists( $propertyId, $constraintParameters ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId( new NumericPropertyId( $propertyId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		$parameters = $constraintParameters[$propertyId];
		if ( count( $parameters ) === 1 &&
			$this->snakDeserializer->deserialize( $parameters[0] ) instanceof PropertyNoValueSnak
		) {
			return [];
		}

		$properties = [];
		foreach ( $parameters as $parameter ) {
			$properties[] = $this->parsePropertyIdParameter( $parameter, $propertyId );
		}
		return $properties;
	}

	/**
	 * @throws ConstraintParameterException
	 */
	private function parseValueOrNoValueParameter( array $snakSerialization, string $parameterId ): ?DataValue {
		$snak = $this->snakDeserializer->deserialize( $snakSerialization );
		if ( $snak instanceof PropertyValueSnak ) {
			return $snak->getDataValue();
		} elseif ( $snak instanceof PropertyNoValueSnak ) {
			return null;
		} else {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-value-or-novalue' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}
	}

	private function parseValueOrNoValueOrNowParameter( array $snakSerialization, string $parameterId ): ?DataValue {
		try {
			return $this->parseValueOrNoValueParameter( $snakSerialization, $parameterId );
		} catch ( ConstraintParameterException $e ) {
			// unknown value means “now”
			return new NowValue();
		}
	}

	/**
	 * Checks whether there is exactly one non-null quantity with the given unit.
	 */
	private function exactlyOneQuantityWithUnit( ?DataValue $min, ?DataValue $max, string $unit ): bool {
		if ( !( $min instanceof UnboundedQuantityValue ) ||
			!( $max instanceof UnboundedQuantityValue )
		) {
			return false;
		}

		return ( $min->getUnit() === $unit ) !== ( $max->getUnit() === $unit );
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $minimumId
	 * @param string $maximumId
	 * @param string $constraintTypeItemId used in error messages
	 * @param string $type 'quantity' or 'time' (can be data type or data value type)
	 *
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return DataValue[] if the parameter is invalid or missing
	 */
	private function parseRangeParameter(
		array $constraintParameters,
		string $minimumId,
		string $maximumId,
		string $constraintTypeItemId,
		string $type
	): array {
		$this->checkError( $constraintParameters );
		if ( !array_key_exists( $minimumId, $constraintParameters ) ||
			!array_key_exists( $maximumId, $constraintParameters )
		) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-range-parameters-needed' ) )
					->withDataValueType( $type )
					->withEntityId( new NumericPropertyId( $minimumId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withEntityId( new NumericPropertyId( $maximumId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
			);
		}

		$this->requireSingleParameter( $constraintParameters, $minimumId );
		$this->requireSingleParameter( $constraintParameters, $maximumId );
		$parseFunction = $type === 'time' ? 'parseValueOrNoValueOrNowParameter' : 'parseValueOrNoValueParameter';
		$min = $this->$parseFunction( $constraintParameters[$minimumId][0], $minimumId );
		$max = $this->$parseFunction( $constraintParameters[$maximumId][0], $maximumId );

		$yearUnit = $this->config->get( 'WBQualityConstraintsYearUnit' );
		if ( $this->exactlyOneQuantityWithUnit( $min, $max, $yearUnit ) ) {
			throw new ConstraintParameterException(
				new ViolationMessage( 'wbqc-violation-message-range-parameters-one-year' )
			);
		}
		if ( $min === null && $max === null ||
			$min !== null && $max !== null && $min->equals( $max ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-range-parameters-same' ) )
					->withEntityId( new NumericPropertyId( $minimumId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withEntityId( new NumericPropertyId( $maximumId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		return [ $min, $max ];
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 *
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return DataValue[] a pair of two data values, either of which may be null to signify an open range
	 */
	public function parseQuantityRangeParameter( array $constraintParameters, string $constraintTypeItemId ): array {
		return $this->parseRangeParameter(
			$constraintParameters,
			$this->config->get( 'WBQualityConstraintsMinimumQuantityId' ),
			$this->config->get( 'WBQualityConstraintsMaximumQuantityId' ),
			$constraintTypeItemId,
			'quantity'
		);
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 *
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return DataValue[] a pair of two data values, either of which may be null to signify an open range
	 */
	public function parseTimeRangeParameter( array $constraintParameters, string $constraintTypeItemId ): array {
		return $this->parseRangeParameter(
			$constraintParameters,
			$this->config->get( 'WBQualityConstraintsMinimumDateId' ),
			$this->config->get( 'WBQualityConstraintsMaximumDateId' ),
			$constraintTypeItemId,
			'time'
		);
	}

	/**
	 * Parse language parameter.
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException
	 * @return string[]
	 */
	public function parseLanguageParameter( array $constraintParameters, string $constraintTypeItemId ): array {
		$this->checkError( $constraintParameters );
		$languagePropertyId = $this->config->get( 'WBQualityConstraintsLanguagePropertyId' );
		if ( !array_key_exists( $languagePropertyId, $constraintParameters ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId( new NumericPropertyId( $languagePropertyId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		$languages = [];
		foreach ( $constraintParameters[$languagePropertyId] as $snak ) {
			$languages[] = $this->parseStringParameter( $snak, $languagePropertyId );
		}
		return $languages;
	}

	/**
	 * Parse a single string parameter.
	 * @throws ConstraintParameterException
	 */
	private function parseStringParameter( array $snakSerialization, string $parameterId ): string {
		$snak = $this->snakDeserializer->deserialize( $snakSerialization );
		$this->requireValueParameter( $snak, $parameterId );
		$value = $snak->getDataValue();
		if ( $value instanceof StringValue ) {
			return $value->getValue();
		} else {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-string' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withDataValue( $value, Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return string
	 */
	public function parseNamespaceParameter( array $constraintParameters, string $constraintTypeItemId ): string {
		$this->checkError( $constraintParameters );
		$namespaceId = $this->config->get( 'WBQualityConstraintsNamespaceId' );
		if ( !array_key_exists( $namespaceId, $constraintParameters ) ) {
			return '';
		}

		$this->requireSingleParameter( $constraintParameters, $namespaceId );
		return $this->parseStringParameter( $constraintParameters[$namespaceId][0], $namespaceId );
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return string
	 */
	public function parseFormatParameter( array $constraintParameters, string $constraintTypeItemId ): string {
		$this->checkError( $constraintParameters );
		$formatId = $this->config->get( 'WBQualityConstraintsFormatAsARegularExpressionId' );
		if ( !array_key_exists( $formatId, $constraintParameters ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-needed' ) )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withEntityId( new NumericPropertyId( $formatId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
			);
		}

		$this->requireSingleParameter( $constraintParameters, $formatId );
		return $this->parseStringParameter( $constraintParameters[$formatId][0], $formatId );
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @throws ConstraintParameterException if the parameter is invalid
	 * @return EntityId[]
	 */
	public function parseExceptionParameter( array $constraintParameters ): array {
		$this->checkError( $constraintParameters );
		$exceptionId = $this->config->get( 'WBQualityConstraintsExceptionToConstraintId' );
		if ( !array_key_exists( $exceptionId, $constraintParameters ) ) {
			return [];
		}

		return array_map(
			function ( $snakSerialization ) use ( $exceptionId ) {
				return $this->parseEntityIdParameter( $snakSerialization, $exceptionId );
			},
			$constraintParameters[$exceptionId]
		);
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @throws ConstraintParameterException if the parameter is invalid
	 * @return string|null 'mandatory', 'suggestion' or null
	 */
	public function parseConstraintStatusParameter( array $constraintParameters ): ?string {
		$this->checkError( $constraintParameters );
		$constraintStatusId = $this->config->get( 'WBQualityConstraintsConstraintStatusId' );
		if ( !array_key_exists( $constraintStatusId, $constraintParameters ) ) {
			return null;
		}

		$mandatoryId = $this->config->get( 'WBQualityConstraintsMandatoryConstraintId' );
		$supportedStatuses = [ new ItemId( $mandatoryId ) ];
		if ( $this->config->get( 'WBQualityConstraintsEnableSuggestionConstraintStatus' ) ) {
			$suggestionId = $this->config->get( 'WBQualityConstraintsSuggestionConstraintId' );
			$supportedStatuses[] = new ItemId( $suggestionId );
		} else {
			$suggestionId = null;
		}

		$this->requireSingleParameter( $constraintParameters, $constraintStatusId );
		$snak = $this->snakDeserializer->deserialize( $constraintParameters[$constraintStatusId][0] );
		$this->requireValueParameter( $snak, $constraintStatusId );
		'@phan-var \Wikibase\DataModel\Snak\PropertyValueSnak $snak';
		$dataValue = $snak->getDataValue();
		'@phan-var EntityIdValue $dataValue';
		$entityId = $dataValue->getEntityId();
		$statusId = $entityId->getSerialization();

		if ( $statusId === $mandatoryId ) {
			return 'mandatory';
		} elseif ( $statusId === $suggestionId ) {
			return 'suggestion';
		} else {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-oneof' ) )
					->withEntityId( new NumericPropertyId( $constraintStatusId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withEntityIdList(
						$supportedStatuses,
						Role::CONSTRAINT_PARAMETER_VALUE
					)
			);
		}
	}

	/**
	 * Require that $dataValue is a {@link MonolingualTextValue}.
	 * @throws ConstraintParameterException
	 */
	private function requireMonolingualTextParameter( DataValue $dataValue, string $parameterId ): void {
		if ( !( $dataValue instanceof MonolingualTextValue ) ) {
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-monolingualtext' ) )
					->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
					->withDataValue( $dataValue, Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
	}

	/**
	 * Parse a series of monolingual text snaks (serialized) into a map from language code to string.
	 *
	 * @throws ConstraintParameterException if invalid snaks are found or a language has multiple texts
	 */
	private function parseMultilingualTextParameter( array $snakSerializations, string $parameterId ): MultilingualTextValue {
		$result = [];

		foreach ( $snakSerializations as $snakSerialization ) {
			$snak = $this->snakDeserializer->deserialize( $snakSerialization );
			$this->requireValueParameter( $snak, $parameterId );

			$value = $snak->getDataValue();
			$this->requireMonolingualTextParameter( $value, $parameterId );
			/** @var MonolingualTextValue $value */
			'@phan-var MonolingualTextValue $value';

			$code = $value->getLanguageCode();
			if ( array_key_exists( $code, $result ) ) {
				throw new ConstraintParameterException(
					( new ViolationMessage( 'wbqc-violation-message-parameter-single-per-language' ) )
						->withEntityId( new NumericPropertyId( $parameterId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
						->withLanguage( $code )
				);
			}

			$result[$code] = $value;
		}

		return new MultilingualTextValue( $result );
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @throws ConstraintParameterException if the parameter is invalid
	 * @return MultilingualTextValue
	 */
	public function parseSyntaxClarificationParameter( array $constraintParameters ): MultilingualTextValue {
		$syntaxClarificationId = $this->config->get( 'WBQualityConstraintsSyntaxClarificationId' );

		if ( !array_key_exists( $syntaxClarificationId, $constraintParameters ) ) {
			return new MultilingualTextValue( [] );
		}

		$syntaxClarifications = $this->parseMultilingualTextParameter(
			$constraintParameters[$syntaxClarificationId],
			$syntaxClarificationId
		);

		return $syntaxClarifications;
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @throws ConstraintParameterException if the parameter is invalid
	 * @return MultilingualTextValue
	 */
	public function parseConstraintClarificationParameter( array $constraintParameters ): MultilingualTextValue {
		$constraintClarificationId = $this->config->get( 'WBQualityConstraintsConstraintClarificationId' );

		if ( !array_key_exists( $constraintClarificationId, $constraintParameters ) ) {
			return new MultilingualTextValue( [] );
		}

		$constraintClarifications = $this->parseMultilingualTextParameter(
			$constraintParameters[$constraintClarificationId],
			$constraintClarificationId
		);

		return $constraintClarifications;
	}

	/**
	 * Parse the constraint scope parameters:
	 * the context types and entity types where the constraint should be checked.
	 * Depending on configuration, this may be the same property ID or two different ones.
	 *
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @param string[] $validContextTypes a list of Context::TYPE_* constants which are valid for this constraint type.
	 * If one of the specified scopes is not in this list, a ConstraintParameterException is thrown.
	 * @param string[] $validEntityTypes a list of entity types which are valid for this constraint type.
	 * If one of the specified entity types is not in this list, a ConstraintParameterException is thrown.
	 * @throws ConstraintParameterException
	 * @return array [ string[]|null $contextTypes, string[]|null $entityTypes ]
	 * the context types and entity types in the parameters (each may be null if not specified)
	 * @suppress PhanTypeArraySuspicious
	 */
	public function parseConstraintScopeParameters(
		array $constraintParameters,
		string $constraintTypeItemId,
		array $validContextTypes,
		array $validEntityTypes
	): array {
		$contextTypeParameterId = $this->config->get( 'WBQualityConstraintsConstraintScopeId' );
		$contextTypeItemIds = $this->parseItemIdsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			false,
			$contextTypeParameterId
		);
		$entityTypeParameterId = $this->config->get( 'WBQualityConstraintsConstraintEntityTypesId' );
		$entityTypeItemIds = $this->parseItemIdsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			false,
			$entityTypeParameterId
		);

		$contextTypeMapping = $this->getConstraintScopeContextTypeMapping();
		$entityTypeMapping = $this->getEntityTypeMapping();

		// these nulls will turn into arrays the first time $contextTypes[] or $entityTypes[] is reached,
		// so they’ll be returned as null iff the parameter was not specified
		$contextTypes = null;
		$entityTypes = null;

		if ( $contextTypeParameterId === $entityTypeParameterId ) {
			$itemIds = $contextTypeItemIds;
			$mapping = $contextTypeMapping + $entityTypeMapping;
			foreach ( $itemIds as $itemId ) {
				$mapped = $this->mapItemId( $itemId, $mapping, $contextTypeParameterId );
				if ( in_array( $mapped, $contextTypeMapping, true ) ) {
					$contextTypes[] = $mapped;
				} else {
					$entityTypes[] = $mapped;
				}
			}
		} else {
			foreach ( $contextTypeItemIds as $contextTypeItemId ) {
				$contextTypes[] = $this->mapItemId(
					$contextTypeItemId,
					$contextTypeMapping,
					$contextTypeParameterId
				);
			}
			foreach ( $entityTypeItemIds as $entityTypeItemId ) {
				$entityTypes[] = $this->mapItemId(
					$entityTypeItemId,
					$entityTypeMapping,
					$entityTypeParameterId
				);
			}
		}

		$this->checkValidScope( $constraintTypeItemId, $contextTypes, $validContextTypes );
		$this->checkValidScope( $constraintTypeItemId, $entityTypes, $validEntityTypes );

		return [ $contextTypes, $entityTypes ];
	}

	private function checkValidScope( string $constraintTypeItemId, ?array $types, array $validTypes ): void {
		$invalidTypes = array_diff( $types ?: [], $validTypes );
		if ( $invalidTypes !== [] ) {
			$invalidType = array_pop( $invalidTypes );
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-invalid-scope' ) )
					->withConstraintScope( $invalidType, Role::CONSTRAINT_PARAMETER_VALUE )
					->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
					->withConstraintScopeList( $validTypes, Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
	}

	/**
	 * Turn an item ID into a full unit string (using the concept URI).
	 */
	private function parseUnitParameter( ItemId $unitId ): string {
		return $this->unitItemConceptBaseUri . $unitId->getSerialization();
	}

	/**
	 * Turn an ItemIdSnakValue into a single unit parameter.
	 *
	 * @throws ConstraintParameterException
	 */
	private function parseUnitItem( ItemIdSnakValue $item ): UnitsParameter {
		switch ( true ) {
			case $item->isValue():
				$unit = $this->parseUnitParameter( $item->getItemId() );
				return new UnitsParameter(
					[ $item->getItemId() ],
					[ UnboundedQuantityValue::newFromNumber( 1, $unit ) ],
					false
				);
			case $item->isSomeValue():
				$qualifierId = $this->config->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
				throw new ConstraintParameterException(
					( new ViolationMessage( 'wbqc-violation-message-parameter-value-or-novalue' ) )
						->withEntityId( new NumericPropertyId( $qualifierId ), Role::CONSTRAINT_PARAMETER_PROPERTY )
				);
			case $item->isNoValue():
				return new UnitsParameter( [], [], true );
		}
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return UnitsParameter
	 */
	public function parseUnitsParameter( array $constraintParameters, string $constraintTypeItemId ): UnitsParameter {
		$items = $this->parseItemsParameter( $constraintParameters, $constraintTypeItemId, true );
		$unitItems = [];
		$unitQuantities = [];
		$unitlessAllowed = false;

		foreach ( $items as $item ) {
			$unit = $this->parseUnitItem( $item );
			$unitItems = array_merge( $unitItems, $unit->getUnitItemIds() );
			$unitQuantities = array_merge( $unitQuantities, $unit->getUnitQuantities() );
			$unitlessAllowed = $unitlessAllowed || $unit->getUnitlessAllowed();
		}

		if ( $unitQuantities === [] && !$unitlessAllowed ) {
			throw new LogicException(
				'The "units" parameter is required, and yet we seem to be missing any allowed unit'
			);
		}

		return new UnitsParameter( $unitItems, $unitQuantities, $unitlessAllowed );
	}

	private function getEntityTypeMapping(): array {
		return [
			$this->config->get( 'WBQualityConstraintsWikibaseItemId' ) => 'item',
			$this->config->get( 'WBQualityConstraintsWikibasePropertyId' ) => 'property',
			$this->config->get( 'WBQualityConstraintsWikibaseLexemeId' ) => 'lexeme',
			$this->config->get( 'WBQualityConstraintsWikibaseFormId' ) => 'form',
			$this->config->get( 'WBQualityConstraintsWikibaseSenseId' ) => 'sense',
			$this->config->get( 'WBQualityConstraintsWikibaseMediaInfoId' ) => 'mediainfo',
		];
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return EntityTypesParameter
	 */
	public function parseEntityTypesParameter( array $constraintParameters, string $constraintTypeItemId ): EntityTypesParameter {
		$entityTypes = [];
		$entityTypeItemIds = [];
		$parameterId = $this->config->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$itemIds = $this->parseItemIdsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			true,
			$parameterId
		);

		$mapping = $this->getEntityTypeMapping();
		foreach ( $itemIds as $itemId ) {
			$entityType = $this->mapItemId( $itemId, $mapping, $parameterId );
			$entityTypes[] = $entityType;
			$entityTypeItemIds[] = $itemId;
		}

		if ( empty( $entityTypes ) ) {
			// @codeCoverageIgnoreStart
			throw new LogicException(
				'The "entity types" parameter is required, ' .
				'and yet we seem to be missing any allowed entity type'
			);
			// @codeCoverageIgnoreEnd
		}

		return new EntityTypesParameter( $entityTypes, $entityTypeItemIds );
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @throws ConstraintParameterException if the parameter is invalid
	 * @return PropertyId[]
	 */
	public function parseSeparatorsParameter( array $constraintParameters ): array {
		$separatorId = $this->config->get( 'WBQualityConstraintsSeparatorId' );

		if ( !array_key_exists( $separatorId, $constraintParameters ) ) {
			return [];
		}

		$parameters = $constraintParameters[$separatorId];
		$separators = [];

		foreach ( $parameters as $parameter ) {
			$separators[] = $this->parsePropertyIdParameter( $parameter, $separatorId );
		}

		return $separators;
	}

	private function getConstraintScopeContextTypeMapping(): array {
		return [
			$this->config->get( 'WBQualityConstraintsConstraintCheckedOnMainValueId' ) => Context::TYPE_STATEMENT,
			$this->config->get( 'WBQualityConstraintsConstraintCheckedOnQualifiersId' ) => Context::TYPE_QUALIFIER,
			$this->config->get( 'WBQualityConstraintsConstraintCheckedOnReferencesId' ) => Context::TYPE_REFERENCE,
		];
	}

	private function getPropertyScopeContextTypeMapping(): array {
		return [
			$this->config->get( 'WBQualityConstraintsAsMainValueId' ) => Context::TYPE_STATEMENT,
			$this->config->get( 'WBQualityConstraintsAsQualifiersId' ) => Context::TYPE_QUALIFIER,
			$this->config->get( 'WBQualityConstraintsAsReferencesId' ) => Context::TYPE_REFERENCE,
		];
	}

	/**
	 * @param array $constraintParameters see {@link \WikibaseQuality\ConstraintReport\Constraint::getConstraintParameters()}
	 * @param string $constraintTypeItemId used in error messages
	 * @throws ConstraintParameterException if the parameter is invalid or missing
	 * @return string[] list of Context::TYPE_* constants
	 */
	public function parsePropertyScopeParameter( array $constraintParameters, string $constraintTypeItemId ): array {
		$contextTypes = [];
		$parameterId = $this->config->get( 'WBQualityConstraintsPropertyScopeId' );
		$itemIds = $this->parseItemIdsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			true,
			$parameterId
		);

		$mapping = $this->getPropertyScopeContextTypeMapping();
		foreach ( $itemIds as $itemId ) {
			$contextTypes[] = $this->mapItemId( $itemId, $mapping, $parameterId );
		}

		if ( empty( $contextTypes ) ) {
			// @codeCoverageIgnoreStart
			throw new LogicException(
				'The "property scope" parameter is required, ' .
				'and yet we seem to be missing any allowed scope'
			);
			// @codeCoverageIgnoreEnd
		}

		return $contextTypes;
	}

}
