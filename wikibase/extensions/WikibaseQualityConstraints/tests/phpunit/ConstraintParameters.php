<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use Config;
use DataValues\DataValue;
use DataValues\MonolingualTextValue;
use DataValues\StringValue;
use DataValues\UnboundedQuantityValue;
use InvalidArgumentException;
use Serializers\Serializer;
use UnexpectedValueException;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\Repo\Parsers\TimeParserFactory;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;

/**
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
trait ConstraintParameters {

	use DefaultConfig;

	/**
	 * @var ConstraintParameterParser
	 */
	private $parser;

	/**
	 * @var Serializer
	 */
	private $snakSerializer;

	/**
	 * @return ConstraintParameterParser
	 */
	public function getConstraintParameterParser() {
		if ( $this->parser === null ) {
			$this->parser = new ConstraintParameterParser(
				self::getDefaultConfig(),
				WikibaseRepo::getBaseDataModelDeserializerFactory(),
				'http://wikibase.example/entity/'
			);
		}

		return $this->parser;
	}

	/**
	 * @return Serializer
	 */
	private function getSnakSerializer() {
		if ( $this->snakSerializer == null ) {
			$this->snakSerializer = WikibaseRepo::getBaseDataModelSerializerFactory()->newSnakSerializer();
		}

		return $this->snakSerializer;
	}

	/**
	 * @param string[] $classIds item ID serializations
	 * @return array[]
	 */
	public function classParameter( array $classIds ) {
		$classParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsClassId' );
		return [
			$classParameterId => array_map(
				function ( $classId ) use ( $classParameterId ) {
					return $this->getSnakSerializer()->serialize(
						new PropertyValueSnak(
							new NumericPropertyId( $classParameterId ),
							new EntityIdValue( new ItemId( $classId ) )
						)
					);
				},
				$classIds
			),
		];
	}

	/**
	 * @param string[] $languageCode
	 * @return array
	 */
	public function languageParameter( array $languageCodes ) {
		$languageParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsLanguagePropertyId' );
		$snaks = [];
		foreach ( $languageCodes as $languageCode ) {
			$snaks[] = $this->getSnakSerializer()->serialize(
				new PropertyValueSnak(
					new NumericPropertyId( $languageParameterId ),
					new StringValue( $languageCode )
				)
			);
		}
		return [
			$languageParameterId => $snaks,
		];
	}

	/**
	 * @param string $relation 'instance', 'subclass', or 'instanceOrSubclass'
	 * @return array[]
	 */
	public function relationParameter( $relation ) {
		$relationParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsRelationId' );
		switch ( $relation ) {
			case 'instance':
				$configKey = 'WBQualityConstraintsInstanceOfRelationId';
				break;
			case 'subclass':
				$configKey = 'WBQualityConstraintsSubclassOfRelationId';
				break;
			case 'instanceOrSubclass':
				$configKey = 'WBQualityConstraintsInstanceOrSubclassOfRelationId';
				break;
			default:
				throw new InvalidArgumentException( '$relation must be instance or subclass' );
		}
		return [
			$relationParameterId => [ $this->getSnakSerializer()->serialize(
				new PropertyValueSnak(
					new NumericPropertyId( $relationParameterId ),
					new EntityIdValue( new ItemId( self::getDefaultConfig()->get( $configKey ) ) )
				)
			) ],
		];
	}

	/**
	 * @param string $propertyId
	 * @return array[]
	 */
	public function propertyParameter( $propertyId ) {
		$propertyParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsPropertyId' );
		return [
			$propertyParameterId => [ $this->getSnakSerializer()->serialize(
				new PropertyValueSnak(
					new NumericPropertyId( $propertyParameterId ),
					new EntityIdValue( new NumericPropertyId( $propertyId ) )
				)
			) ],
		];
	}

	/**
	 * @param string[] $properties property ID serializations
	 * @return array[]
	 */
	public function propertiesParameter( array $properties ) {
		$propertyParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsPropertyId' );
		return [
			$propertyParameterId => array_map(
				function ( $property ) use ( $propertyParameterId ) {
					$value = new EntityIdValue( new NumericPropertyId( $property ) );
					$snak = new PropertyValueSnak( new NumericPropertyId( $propertyParameterId ), $value );
					return $this->getSnakSerializer()->serialize( $snak );
				},
				$properties
			),
		];
	}

	/**
	 * @param (string|Snak)[] $items item ID serializations or snaks
	 * @return array[]
	 */
	public function itemsParameter( array $items ) {
		$qualifierParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		return [
			$qualifierParameterId => array_map(
				function ( $item ) use ( $qualifierParameterId ) {
					if ( $item instanceof Snak ) {
						$snak = $item;
					} else {
						$value = new EntityIdValue( new ItemId( $item ) );
						$snak = new PropertyValueSnak( new NumericPropertyId( $qualifierParameterId ), $value );
					}
					return $this->getSnakSerializer()->serialize( $snak );
				},
				$items
			),
		];
	}

	/**
	 * Convert an abbreviated value for a range endpoint
	 * to a full snak for range constraint parameters.
	 * A numeric argument means a numeric endpoint,
	 * 'now' corresponds to a somevalue snak,
	 * any other string is parsed as a time value,
	 * a DataValue is used directly,
	 * and null corresponds to a novalue snak (open-ended range).
	 *
	 * @param DataValue|int|float|string|null $value
	 * @param string $property property ID serialization
	 * @return Snak
	 */
	private function rangeEndpoint( $value, $property ) {
		$propertyId = new NumericPropertyId( $property );
		if ( $value === null ) {
			return new PropertyNoValueSnak( $propertyId );
		} else {
			if ( is_string( $value ) ) {
				if ( $value === 'now' ) {
					return new PropertySomeValueSnak( $propertyId );
				}
				$timeParser = ( new TimeParserFactory() )->getTimeParser();
				$value = $timeParser->parse( $value );
			} elseif ( is_numeric( $value ) ) {
				$value = UnboundedQuantityValue::newFromNumber( $value );
			}
			return new PropertyValueSnak( $propertyId, $value );
		}
	}

	/**
	 * @param string $type 'quantity' or 'time'
	 * @param DataValue|int|float|string|null $min lower boundary, see rangeEndpoint() for details
	 * @param DataValue|int|float|string|null $max upper boundary, see rangeEndpoint() for details
	 * @return array[]
	 */
	public function rangeParameter( $type, $min, $max ) {
		$configKey = $type === 'quantity' ? 'Quantity' : 'Date';
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimum' . $configKey . 'Id' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximum' . $configKey . 'Id' );
		$minimumSnak = $this->rangeEndpoint( $min, $minimumId );
		$maximumSnak = $this->rangeEndpoint( $max, $maximumId );
		$snakSerializer = $this->getSnakSerializer();
		return [
			$minimumId => [ $snakSerializer->serialize( $minimumSnak ) ],
			$maximumId => [ $snakSerializer->serialize( $maximumSnak ) ],
		];
	}

	/**
	 * @param string $namespace
	 * @return array[]
	 */
	public function namespaceParameter( $namespace ) {
		$namespaceId = self::getDefaultConfig()->get( 'WBQualityConstraintsNamespaceId' );
		$value = new StringValue( $namespace );
		$snak = new PropertyValueSnak( new NumericPropertyId( $namespaceId ), $value );
		return [ $namespaceId => [ $this->getSnakSerializer()->serialize( $snak ) ] ];
	}

	/**
	 * @param string $format
	 * @return array[]
	 */
	public function formatParameter( $format ) {
		$formatId = self::getDefaultConfig()->get( 'WBQualityConstraintsFormatAsARegularExpressionId' );
		$value = new StringValue( $format );
		$snak = new PropertyValueSnak( new NumericPropertyId( $formatId ), $value );
		return [ $formatId => [ $this->getSnakSerializer()->serialize( $snak ) ] ];
	}

	/**
	 * @param string $languageCode
	 * @param string $syntaxClarification
	 * @return array[]
	 */
	public function syntaxClarificationParameter( $languageCode, $syntaxClarification ) {
		$syntaxClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$value = new MonolingualTextValue( $languageCode, $syntaxClarificationId );
		$snak = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value );
		return [ $syntaxClarificationId => [ $this->getSnakSerializer()->serialize( $snak ) ] ];
	}

	/**
	 * @param string[] $exceptions item ID serializations (other entity types currently not supported)
	 * @return array[]
	 */
	public function exceptionsParameter( $exceptions ) {
		$exceptionId = self::getDefaultConfig()->get( 'WBQualityConstraintsExceptionToConstraintId' );
		return [ $exceptionId => array_map(
			function ( $exception ) use ( $exceptionId ) {
				$value = new EntityIdValue( new ItemId( $exception ) );
				$snak = new PropertyValueSnak( new NumericPropertyId( $exceptionId ), $value );
				return $this->getSnakSerializer()->serialize( $snak );
			},
			$exceptions
		) ];
	}

	/**
	 * @param string $status 'mandatory', 'suggestion', or 'invalid'
	 * @return array[]
	 */
	public function statusParameter( $status ) {
		$statusParameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintStatusId' );
		switch ( $status ) {
			case 'mandatory':
				$configKey = 'WBQualityConstraintsMandatoryConstraintId';
				break;
			case 'suggestion':
				$configKey = 'WBQualityConstraintsSuggestionConstraintId';
				break;
			case 'invalid':
				$configKey = 'WBQualityConstraintsAsMainValueId'; // unrelated, invalid as a constraint status
				break;
			default:
				throw new InvalidArgumentException( '$status must be mandatory or suggestion' );
		}
		return [
			$statusParameterId => [ $this->getSnakSerializer()->serialize(
				new PropertyValueSnak(
					new NumericPropertyId( $statusParameterId ),
					new EntityIdValue( new ItemId( self::getDefaultConfig()->get( $configKey ) ) )
				)
			) ],
		];
	}

	/**
	 * @param string $languageCode
	 * @param string $constraintClarification
	 * @return array[]
	 */
	public function constraintClarificationParameter( $languageCode, $constraintClarification ) {
		$constraintClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintClarificationId' );
		$value = new MonolingualTextValue( $languageCode, $constraintClarification );
		$snak = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value );
		return [ $constraintClarificationId => [ $this->getSnakSerializer()->serialize( $snak ) ] ];
	}

	/**
	 * @param string[] $contextTypes Context::TYPE_* constants
	 * @param string[] $entityTypes
	 * @return array
	 */
	public function constraintScopeParameter( array $contextTypes, array $entityTypes = [] ) {
		$config = self::getDefaultConfig();
		$constraintScopeParameterId = $config->get( 'WBQualityConstraintsConstraintScopeId' );
		$contextTypeItemIds = [];
		foreach ( $contextTypes as $contextType ) {
			$contextTypeItemIds[] = $this->contextTypeToItemId( $config, $contextType );
		}
		// ConstraintParameterParser allows these to be different,
		// but they default to the same property,
		// so we assert that here to keep this function simpler
		$this->assertSame( $constraintScopeParameterId, $config->get( 'WBQualityConstraintsConstraintEntityTypesId' ) );
		$entityTypeItemIds = [];
		foreach ( $entityTypes as $entityType ) {
			$entityTypeItemIds[] = $this->entityTypeToItemId( $config, $entityType );
		}
		return [ $constraintScopeParameterId => array_map(
			function ( $itemId ) use ( $constraintScopeParameterId ) {
				return $this->getSnakSerializer()->serialize(
					new PropertyValueSnak(
						new NumericPropertyId( $constraintScopeParameterId ),
						new EntityIdValue( new ItemId( $itemId ) )
					)
				);
			},
			array_merge( $contextTypeItemIds, $entityTypeItemIds )
		) ];
	}

	private function contextTypeToItemId( Config $config, string $contextType ): string {
		switch ( $contextType ) {
			case Context::TYPE_STATEMENT:
				return $config->get( 'WBQualityConstraintsConstraintCheckedOnMainValueId' );
			case Context::TYPE_QUALIFIER:
				return $config->get( 'WBQualityConstraintsConstraintCheckedOnQualifiersId' );
			case Context::TYPE_REFERENCE:
				return $config->get( 'WBQualityConstraintsConstraintCheckedOnReferencesId' );
			default:
				$this->fail( 'unknown context type ' . $contextType );
		}
	}

	private function entityTypeToItemId( Config $config, string $entityType ): string {
		switch ( $entityType ) {
			case 'item':
				return $config->get( 'WBQualityConstraintsWikibaseItemId' );
			case 'property':
				return $config->get( 'WBQualityConstraintsWikibasePropertyId' );
			case 'lexeme':
			case 'form':
			case 'sense':
			case 'mediainfo':
				throw new UnexpectedValueException(
					"support for entity type $entityType omitted, add when needed"
				);
			default:
				$this->assertTrue( false, 'unknown entity type ' . $entityType );
		}
	}

	public function separatorsParameter( array $separators ) {
		$separatorId = self::getDefaultConfig()->get( 'WBQualityConstraintsSeparatorId' );
		return [
			$separatorId => array_map(
				function ( $separator ) use ( $separatorId ) {
					$value = new EntityIdValue( new NumericPropertyId( $separator ) );
					$snak = new PropertyValueSnak( new NumericPropertyId( $separatorId ), $value );
					return $this->getSnakSerializer()->serialize( $snak );
				},
				$separators
			),
		];
	}

	public function propertyScopeParameter( array $contextTypes ) {
		$config = self::getDefaultConfig();
		$parameterId = $config->get( 'WBQualityConstraintsPropertyScopeId' );
		return [
			$parameterId => array_map(
				function ( $contextType ) use ( $config, $parameterId ) {
					switch ( $contextType ) {
						case Context::TYPE_STATEMENT:
							$itemId = $config->get( 'WBQualityConstraintsAsMainValueId' );
							break;
						case Context::TYPE_QUALIFIER:
							$itemId = $config->get( 'WBQualityConstraintsAsQualifiersId' );
							break;
						case Context::TYPE_REFERENCE:
							$itemId = $config->get( 'WBQualityConstraintsAsReferencesId' );
							break;
						default:
							$this->fail( 'unknown context type ' . $contextType );
					}
					$value = new EntityIdValue( new ItemId( $itemId ) );
					$snak = new PropertyValueSnak( new NumericPropertyId( $parameterId ), $value );
					return $this->getSnakSerializer()->serialize( $snak );
				},
				$contextTypes
			),
		];
	}

}
