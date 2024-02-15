<?php

namespace WikibaseQuality\ConstraintReport\Tests\Helper;

use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\StringValue;
use DataValues\TimeValue;
use DataValues\UnboundedQuantityValue;
use HashConfig;
use MultiConfig;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\NowValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ConstraintParameterParserTest extends \MediaWikiLangTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var Constraint
	 */
	private $constraint;

	protected function setUp(): void {
		parent::setUp();
		$this->constraint = new Constraint( 'constraint ID', new NumericPropertyId( 'P1' ), 'constraint type Q-ID', [] );
	}

	/**
	 * @param string $itemId
	 * @return array
	 */
	private function serializeItemId( $itemId ) {
		return $this->getSnakSerializer()->serialize(
			new PropertyValueSnak(
				new NumericPropertyId( 'P1' ),
				new EntityIdValue( new ItemId( $itemId ) )
			)
		);
	}

	/**
	 * @param string $propertyId
	 * @return array
	 */
	private function serializePropertyId( $propertyId ) {
		return $this->getSnakSerializer()->serialize(
			new PropertyValueSnak(
				new NumericPropertyId( 'P1' ),
				new EntityIdValue( new NumericPropertyId( $propertyId ) )
			)
		);
	}

	/**
	 * @param string $method
	 * @param array $arguments
	 * @param string $messageKey
	 * @see \WikibaseQuality\ConstraintReport\Tests\ResultAssertions::assertViolation
	 */
	private function assertThrowsConstraintParameterException( $method, array $arguments, $messageKey ) {
		try {
			call_user_func_array( [ $this->getConstraintParameterParser(), $method ], $arguments );
			$this->fail(
				"$method should have thrown a ConstraintParameterException with message ⧼{$messageKey}⧽." );
		} catch ( ConstraintParameterException $exception ) {
			$checkResult = new CheckResult(
				new MainSnakContext(
					new Item( new ItemId( 'Q1' ) ),
					new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
				),
				$this->constraint,
				CheckResult::STATUS_VIOLATION,
				$exception->getViolationMessage()
			);
			$this->assertViolation( $checkResult, $messageKey );
		}
	}

# region parseClassParameter
	public function testParseClassParameter() {
		$config = self::getDefaultConfig();
		$classId = $config->get( 'WBQualityConstraintsClassId' );
		$parsed = $this->getConstraintParameterParser()->parseClassParameter(
			[ $classId => [ $this->serializeItemId( 'Q100' ), $this->serializeItemId( 'Q101' ) ] ],
			'Q1'
		);
		$this->assertSame( [ 'Q100', 'Q101' ], $parsed );
	}

	public function testParseClassParameter_Missing() {
		$this->assertThrowsConstraintParameterException(
			'parseClassParameter',
			[ [], 'Q21503250' ],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParseClassParameter_NoValue() {
		$config = self::getDefaultConfig();
		$classId = $config->get( 'WBQualityConstraintsClassId' );
		$this->assertThrowsConstraintParameterException(
			'parseClassParameter',
			[
				[
					$classId => [
						$this->getSnakSerializer()->serialize( new PropertyNoValueSnak( new NumericPropertyId( $classId ) ) ),
					],
				],
				'Q21503250',
			],
			'wbqc-violation-message-parameter-value'
		);
	}

	public function testParseClassParameter_StringValue() {
		$config = self::getDefaultConfig();
		$classId = $config->get( 'WBQualityConstraintsClassId' );
		$this->assertThrowsConstraintParameterException(
			'parseClassParameter',
			[
				[
					$classId => [
						$this->getSnakSerializer()->serialize( new PropertyValueSnak(
							new NumericPropertyId( $classId ),
							new StringValue( 'Q100' )
						) ),
					],
				],
				'Q21503250',
			],
			'wbqc-violation-message-parameter-entity'
		);
	}

# endregion

# region parseRelationParameter
	public function testParseRelationParameter() {
		$config = self::getDefaultConfig();
		$relationId = $config->get( 'WBQualityConstraintsRelationId' );
		$instanceOfId = $config->get( 'WBQualityConstraintsInstanceOfRelationId' );
		$parsed = $this->getConstraintParameterParser()->parseRelationParameter(
			[ $relationId => [ $this->serializeItemId( $instanceOfId ) ] ],
			'Q21503250'
		);
		$this->assertSame( 'instance', $parsed );
	}

	public function testParseRelationParameter_Missing() {
		$this->assertThrowsConstraintParameterException(
			'parseRelationParameter',
			[ [], 'Q21503250' ],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParseRelationParameter_NoValue() {
		$config = self::getDefaultConfig();
		$relationId = $config->get( 'WBQualityConstraintsRelationId' );
		$this->assertThrowsConstraintParameterException(
			'parseRelationParameter',
			[
				[
					$relationId => [
						$this->getSnakSerializer()->serialize( new PropertyNoValueSnak( new NumericPropertyId( $relationId ) ) ),
					],
				],
				'Q21503250',
			],
			'wbqc-violation-message-parameter-value'
		);
	}

	public function testParseRelationParameter_StringValue() {
		$config = self::getDefaultConfig();
		$relationId = $config->get( 'WBQualityConstraintsRelationId' );
		$this->assertThrowsConstraintParameterException(
			'parseRelationParameter',
			[
				[
					$relationId => [
						$this->getSnakSerializer()->serialize( new PropertyValueSnak(
							new NumericPropertyId( $relationId ),
							new StringValue( 'instance' )
						) ),
					],
				],
				'Q21503250',
			],
			'wbqc-violation-message-parameter-entity'
		);
	}

	public function testParseRelationParameter_MultiValue() {
		$config = self::getDefaultConfig();
		$relationId = $config->get( 'WBQualityConstraintsRelationId' );
		$instanceOfId = $config->get( 'WBQualityConstraintsInstanceOfRelationId' );
		$subclassOfId = $config->get( 'WBQualityConstraintsSubclassOfRelationId' );
		$this->assertThrowsConstraintParameterException(
			'parseRelationParameter',
			[
				[
					$relationId => [
						$this->serializeItemId( $instanceOfId ),
						$this->serializeItemId( $subclassOfId ),
					],
				],
				'Q21503250',
			],
			'wbqc-violation-message-parameter-single'
		);
	}

	public function testParseRelationParameter_WrongValue() {
		$config = self::getDefaultConfig();
		$relationId = $config->get( 'WBQualityConstraintsRelationId' );
		$this->assertThrowsConstraintParameterException(
			'parseRelationParameter',
			[
				[
					$relationId => [ $this->serializeItemId( 'Q1' ) ],
				],
				'Q21503250',
			],
			'wbqc-violation-message-parameter-oneof'
		);
	}

# endregion

# region parsePropertyParameter
	public function testParsePropertyParameter() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$parsed = $this->getConstraintParameterParser()->parsePropertyParameter(
			[ $propertyId => [ $this->serializePropertyId( 'P100' ) ] ],
			'Q21510856'
		);

		$this->assertEquals( new NumericPropertyId( 'P100' ), $parsed );
	}

	public function testParsePropertyParameter_Missing() {
		$this->assertThrowsConstraintParameterException(
			'parsePropertyParameter',
			[ [], 'Q21510856' ],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParsePropertyParameter_NoValue() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$this->assertThrowsConstraintParameterException(
			'parsePropertyParameter',
			[
				[
					$propertyId => [
						$this->getSnakSerializer()->serialize( new PropertyNoValueSnak( new NumericPropertyId( $propertyId ) ) ),
					],
				],
				'Q21510856',
			],
			'wbqc-violation-message-parameter-value'
		);
	}

	public function testParsePropertyParameter_StringValue() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$this->assertThrowsConstraintParameterException(
			'parsePropertyParameter',
			[
				[
					$propertyId => [
						$this->getSnakSerializer()->serialize( new PropertyValueSnak(
							new NumericPropertyId( $propertyId ),
							new StringValue( 'P1' )
						) ),
					],
				],
				'Q21510856',
			],
			'wbqc-violation-message-parameter-property'
		);
	}

	public function testParsePropertyParameter_ItemId() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$this->assertThrowsConstraintParameterException(
			'parsePropertyParameter',
			[
				[
					$propertyId => [
						$this->serializeItemId( 'Q100' ),
					],
				],
				'Q21510856',
			],
			'wbqc-violation-message-parameter-property'
		);
	}

	public function testParsePropertyParameter_MultiValue() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$this->assertThrowsConstraintParameterException(
			'parsePropertyParameter',
			[
				[
					$propertyId => [
						$this->serializePropertyId( 'P100' ),
						$this->serializePropertyId( 'P101' ),
					],
				],
				'Q21510856',
			],
			'wbqc-violation-message-parameter-single'
		);
	}

# endregion

# region parseItemsParameter
	public function testParseItemsParameter() {
		$config = self::getDefaultConfig();
		$qualifierId = $config->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$parsed = $this->getConstraintParameterParser()->parseItemsParameter(
			[
				$qualifierId => [
					$this->serializeItemId( 'Q100' ),
					$this->serializeItemId( 'Q101' ),
					$this->getSnakSerializer()->serialize( new PropertySomeValueSnak( new NumericPropertyId( 'P1' ) ) ),
					$this->getSnakSerializer()->serialize( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) ),
				],
			],
			'Q21510859',
			false
		);
		$expected = [
			ItemIdSnakValue::fromItemId( new ItemId( 'Q100' ) ),
			ItemIdSnakValue::fromItemId( new ItemId( 'Q101' ) ),
			ItemIdSnakValue::someValue(),
			ItemIdSnakValue::noValue(),
		];
		$this->assertEquals( $expected, $parsed );
	}

	public function testParseItemsParameter_Required() {
		$this->assertThrowsConstraintParameterException(
			'parseItemsParameter',
			[
				[], 'Q21510859', true,
			],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParseItemsParameter_NotRequired() {
		$parsed = $this->getConstraintParameterParser()->parseItemsParameter( [], 'Q21510859', false );
		$this->assertEquals( [], $parsed );
	}

	public function testParseItemsParameter_StringValue() {
		$config = self::getDefaultConfig();
		$qualifierId = $config->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$this->assertThrowsConstraintParameterException(
			'parseItemsParameter',
			[
				[
					$qualifierId => [
						$this->getSnakSerializer()->serialize( new PropertyValueSnak(
							new NumericPropertyId( 'P1' ),
							new StringValue( 'Q100' )
						) ),
					],
				],
				'Q21510859',
				true,
			],
			'wbqc-violation-message-parameter-item'
		);
	}

	public function testParseItemsParameter_PropertyId() {
		$config = self::getDefaultConfig();
		$qualifierId = $config->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$this->assertThrowsConstraintParameterException(
			'parseItemsParameter',
			[
				[
					$qualifierId => [
						$this->getSnakSerializer()->serialize( new PropertyValueSnak(
							new NumericPropertyId( 'P1' ),
							new EntityIdValue( new NumericPropertyId( 'P100' ) )
						) ),
					],
				],
				'Q21510859',
				true,
			],
			'wbqc-violation-message-parameter-item'
		);
	}

# endregion

# region parsePropertiesParameter
	public function testParsePropertiesParameter() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$parsed = $this->getConstraintParameterParser()->parsePropertiesParameter(
			[ $propertyId => [ $this->serializePropertyId( 'P100' ), $this->serializePropertyId( 'P101' ) ] ],
			'Q21510851'
		);

		$this->assertEquals( [ new NumericPropertyId( 'P100' ), new NumericPropertyId( 'P101' ) ], $parsed );
	}

	public function testParsePropertiesParameter_Missing() {
		$this->assertThrowsConstraintParameterException(
			'parsePropertiesParameter',
			[ [], 'Q21510851' ],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParsePropertiesParameter_NoValue() {
		$config = self::getDefaultConfig();
		$propertyId = $config->get( 'WBQualityConstraintsPropertyId' );
		$parsed = $this->getConstraintParameterParser()->parsePropertiesParameter(
			[ $propertyId => [
				$this->getSnakSerializer()->serialize( new PropertyNoValueSnak( new NumericPropertyId( $propertyId ) ) ),
			] ],
			'Q21510851'
		);
		$this->assertEquals( [], $parsed );
	}

# endregion

# region parseQuantityRangeParameter
	public function testParseQuantityRange_Bounded() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$propertyId = new NumericPropertyId( 'P1' );
		$min = UnboundedQuantityValue::newFromNumber( 13.37 );
		$max = UnboundedQuantityValue::newFromNumber( 42 );

		$parsed = $this->getConstraintParameterParser()->parseQuantityRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $min ) ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $max ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ $min, $max ], $parsed );
	}

	public function testParseQuantityRange_LeftOpen() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$propertyId = new NumericPropertyId( 'P1' );
		$max = UnboundedQuantityValue::newFromNumber( 42 );

		$parsed = $this->getConstraintParameterParser()->parseQuantityRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyNoValueSnak( $propertyId ) ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $max ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ null, $max ], $parsed );
	}

	public function testParseQuantityRange_RightOpen() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$propertyId = new NumericPropertyId( 'P1' );
		$min = UnboundedQuantityValue::newFromNumber( 13.37 );

		$parsed = $this->getConstraintParameterParser()->parseQuantityRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $min ) ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyNoValueSnak( $propertyId ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ $min, null ], $parsed );
	}

	public function testParseQuantityRange_FullyOpen() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$propertyId = new NumericPropertyId( 'P1' );

		$this->assertThrowsConstraintParameterException(
			'parseQuantityRangeParameter',
			[
				[
					$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyNoValueSnak( $propertyId ) ) ],
					$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyNoValueSnak( $propertyId ) ) ],
				],
				'Q21510860',
			],
			'wbqc-violation-message-range-parameters-same'
		);
	}

	public function testParseQuantityRange_SomeValue() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$propertyId = new NumericPropertyId( 'P1' );

		$this->assertThrowsConstraintParameterException(
			'parseQuantityRangeParameter',
			[
				[
					$minimumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
					$maximumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
				],
				'Q21510860',
			],
			'wbqc-violation-message-parameter-value-or-novalue'
		);
	}

	public function testParseQuantityRange_Same() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$propertyId = new NumericPropertyId( 'P1' );
		$quantity = UnboundedQuantityValue::newFromNumber( 13.37 );

		$this->assertThrowsConstraintParameterException(
			'parseQuantityRangeParameter',
			[
				[
					$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $quantity ) ) ],
					$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $quantity ) ) ],
				],
				'Q21510860',
			],
			'wbqc-violation-message-range-parameters-same'
		);
	}

	public function testParseQuantityRange_OneYear() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$yearUnit = $config->get( 'WBQualityConstraintsYearUnit' );
		$min = UnboundedQuantityValue::newFromNumber( 0, 'other unit than ' . $yearUnit );
		$max = UnboundedQuantityValue::newFromNumber( 150, $yearUnit );
		$minSnak = new PropertyValueSnak( new NumericPropertyId( $minimumId ), $min );
		$maxSnak = new PropertyValueSnak( new NumericPropertyId( $maximumId ), $max );

		$this->assertThrowsConstraintParameterException(
			'parseQuantityRangeParameter',
			[
				[
					$minimumId => [ $this->getSnakSerializer()->serialize( $minSnak ) ],
					$maximumId => [ $this->getSnakSerializer()->serialize( $maxSnak ) ],
				],
				'Q21510860',
			],
			'wbqc-violation-message-range-parameters-one-year'
		);
	}

	public function testParseQuantityRange_OneYear_LeftOpen() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumQuantityId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumQuantityId' );
		$yearUnit = $config->get( 'WBQualityConstraintsYearUnit' );
		$max = UnboundedQuantityValue::newFromNumber( 150, $yearUnit );
		$minSnak = new PropertyNoValueSnak( new NumericPropertyId( $minimumId ) );
		$maxSnak = new PropertyValueSnak( new NumericPropertyId( $maximumId ), $max );

		$parsed = $this->getConstraintParameterParser()->parseQuantityRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( $minSnak ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( $maxSnak ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ null, $max ], $parsed );
	}

# endregion

# region parseTimeRangeParameter
	public function testParseTimeRange_Bounded() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumDateId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumDateId' );
		$propertyId = new NumericPropertyId( 'P1' );
		$calendar = TimeValue::CALENDAR_GREGORIAN;
		$min = new TimeValue( '+1789-05-08T00:00:00Z', 0, 0, 0, TimeValue::PRECISION_YEAR, $calendar );
		$max = new TimeValue( '+1955-02-05T00:00:00Z', 0, 0, 0, TimeValue::PRECISION_YEAR, $calendar );

		$parsed = $this->getConstraintParameterParser()->parseTimeRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $min ) ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyValueSnak( $propertyId, $max ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ $min, $max ], $parsed );
	}

	public function testParseTimeRange_Past() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumDateId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumDateId' );
		$propertyId = new NumericPropertyId( 'P1' );

		$parsed = $this->getConstraintParameterParser()->parseTimeRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( new PropertyNoValueSnak( $propertyId ) ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ null, new NowValue() ], $parsed );
	}

	public function testParseTimeRange_Future() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumDateId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumDateId' );
		$propertyId = new NumericPropertyId( 'P1' );

		$parsed = $this->getConstraintParameterParser()->parseTimeRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertyNoValueSnak( $propertyId ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ new NowValue(), null ], $parsed );
	}

	public function testParseTimeRange_BothNow() {
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumDateId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumDateId' );
		$propertyId = new NumericPropertyId( 'P1' );

		$this->assertThrowsConstraintParameterException(
			'parseTimeRangeParameter',
			[
				[
					$minimumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
					$maximumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
				],
				'Q21510860',
			],
			'wbqc-violation-message-range-parameters-same'
		);
	}

	public function testParseTimeRange_Wikidata() {
		// range: from the inception of Wikidata until now
		// (NowValue uses the same date internally, but that should not result in a “same range endpoints” error)
		$config = self::getDefaultConfig();
		$minimumId = $config->get( 'WBQualityConstraintsMinimumDateId' );
		$maximumId = $config->get( 'WBQualityConstraintsMaximumDateId' );
		$propertyId = new NumericPropertyId( 'P1' );
		$wikidataInception = new TimeValue(
			'+2012-10-29T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_SECOND,
			TimeValue::CALENDAR_GREGORIAN
		);

		$parsed = $this->getConstraintParameterParser()->parseTimeRangeParameter(
			[
				$minimumId => [ $this->getSnakSerializer()->serialize(
					new PropertyValueSnak( $propertyId, $wikidataInception )
				) ],
				$maximumId => [ $this->getSnakSerializer()->serialize( new PropertySomeValueSnak( $propertyId ) ) ],
			],
			'Q21510860'
		);

		$this->assertEquals( [ $wikidataInception, new NowValue() ], $parsed );
	}

# endregion

	public function testParseRange_MissingParameters() {
		foreach ( [ 'parseQuantityRangeParameter', 'parseTimeRangeParameter' ] as $method ) {
			$this->assertThrowsConstraintParameterException(
				$method,
				[
					[],
					'Q21510860',
				],
				'wbqc-violation-message-range-parameters-needed'
			);
		}
	}

# region parseNamespaceParameter
	public function testParseNamespaceParameter() {
		$namespaceId = self::getDefaultConfig()->get( 'WBQualityConstraintsNamespaceId' );
		$value = new StringValue( 'File' );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value );

		$parsed = $this->getConstraintParameterParser()->parseNamespaceParameter(
			[ $namespaceId => [ $this->getSnakSerializer()->serialize( $snak ) ] ],
			'Q21510852'
		);

		$this->assertSame( 'File', $parsed );
	}

	public function testParseNamespaceParameter_Missing() {
		$parsed = $this->getConstraintParameterParser()->parseNamespaceParameter(
			[],
			'Q21510852'
		);

		$this->assertSame( '', $parsed );
	}

	public function testParseNamespaceParameter_ItemId() {
		$namespaceId = self::getDefaultConfig()->get( 'WBQualityConstraintsNamespaceId' );
		$value = new EntityIdValue( new ItemId( 'Q1' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value );

		$this->assertThrowsConstraintParameterException(
			'parseNamespaceParameter',
			[
				[ $namespaceId => [ $this->getSnakSerializer()->serialize( $snak ) ] ],
				'Q21510852',
			],
			'wbqc-violation-message-parameter-string'
		);
	}

	public function testParseNamespaceParameter_Multiple() {
		$namespaceId = self::getDefaultConfig()->get( 'WBQualityConstraintsNamespaceId' );
		$value1 = new StringValue( 'File' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value1 );
		$value2 = new StringValue( 'Category' );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value2 );

		$this->assertThrowsConstraintParameterException(
			'parseNamespaceParameter',
			[
				[ $namespaceId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
				] ],
				'Q21510852',
			],
			'wbqc-violation-message-parameter-single'
		);
	}

# endregion

# region parseFormatParameter
	public function testParseFormatParameter() {
		$formatId = self::getDefaultConfig()->get( 'WBQualityConstraintsFormatAsARegularExpressionId' );
		$value = new StringValue( '\d\.(\d{1,2}|-{1})\.(\d{1,2}|-{1})\.(\d{1,3}|-{1})' );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value );

		$parsed = $this->getConstraintParameterParser()->parseFormatParameter(
			[ $formatId => [ $this->getSnakSerializer()->serialize( $snak ) ] ],
			'Q21502404'
		);

		$this->assertSame( '\d\.(\d{1,2}|-{1})\.(\d{1,2}|-{1})\.(\d{1,3}|-{1})', $parsed );
	}

	public function testParseFormatParameter_Missing() {
		$this->assertThrowsConstraintParameterException(
			'parseFormatParameter',
			[
				[],
				'Q21502404',
			],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParseFormatParameter_ItemId() {
		$formatId = self::getDefaultConfig()->get( 'WBQualityConstraintsFormatAsARegularExpressionId' );
		$value = new EntityIdValue( new ItemId( 'Q1' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value );

		$this->assertThrowsConstraintParameterException(
			'parseFormatParameter',
			[
				[ $formatId => [ $this->getSnakSerializer()->serialize( $snak ) ] ],
				'Q21502404',
			],
			'wbqc-violation-message-parameter-string'
		);
	}

	public function testParseFormatParameter_Multiple() {
		$formatId = self::getDefaultConfig()->get( 'WBQualityConstraintsFormatAsARegularExpressionId' );
		$value1 = new StringValue( '\d\.(\d{1,2}|-{1})\.(\d{1,2}|-{1})\.(\d{1,3}|-{1})' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value1 );
		$value2 = new StringValue( '\d+' );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $value2 );

		$this->assertThrowsConstraintParameterException(
			'parseFormatParameter',
			[
				[ $formatId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
				] ],
				'Q21502404',
			],
			'wbqc-violation-message-parameter-single'
		);
	}

# endregion

# region parseExceptionParameter
	public function testParseExceptionParameter() {
		$exceptionId = self::getDefaultConfig()->get( 'WBQualityConstraintsExceptionToConstraintId' );
		$entityId1 = new ItemId( 'Q100' );
		$entityId2 = new NumericPropertyId( 'P100' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( $exceptionId ), new EntityIdValue( $entityId1 ) );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( $exceptionId ), new EntityIdValue( $entityId2 ) );

		$parsed = $this->getConstraintParameterParser()->parseExceptionParameter(
			[ $exceptionId => [
				$this->getSnakSerializer()->serialize( $snak1 ),
				$this->getSnakSerializer()->serialize( $snak2 ),
			] ]
		);

		$this->assertEquals( [ $entityId1, $entityId2 ], $parsed );
	}

	public function testParseExceptionParameter_Missing() {
		$parsed = $this->getConstraintParameterParser()->parseExceptionParameter(
			[]
		);

		$this->assertEquals( [], $parsed );
	}

	public function testParseExceptionParameter_String() {
		$exceptionId = self::getDefaultConfig()->get( 'WBQualityConstraintsExceptionToConstraintId' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $exceptionId ), new StringValue( 'Q100' ) );

		$this->assertThrowsConstraintParameterException(
			'parseExceptionParameter',
			[ [ $exceptionId => [ $this->getSnakSerializer()->serialize( $snak ) ] ] ],
			'wbqc-violation-message-parameter-entity'
		);
	}

	public function testParseExceptionParameter_TooLong() {
		$this->assertThrowsConstraintParameterException(
			'parseExceptionParameter',
			[ [ '@error' => [ 'toolong' => true ] ] ],
			'wbqc-violation-message-parameters-error-toolong'
		);
	}

# endregion

# region parseConstraintStatusParameter
	public function testParseConstraintStatusParameter_mandatory() {
		$constraintStatusId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintStatusId' );
		$mandatoryId = self::getDefaultConfig()->get( 'WBQualityConstraintsMandatoryConstraintId' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( $constraintStatusId ),
			new EntityIdValue( new ItemId( $mandatoryId ) )
		);

		$parsed = $this->getConstraintParameterParser()->parseConstraintStatusParameter(
			[ $constraintStatusId => [ $this->getSnakSerializer()->serialize( $snak ) ] ]
		);

		$this->assertSame( 'mandatory', $parsed );
	}

	public function testParseConstraintStatusParameter_suggestion_disabled() {
		$constraintStatusId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintStatusId' );
		$suggestionId = self::getDefaultConfig()->get( 'WBQualityConstraintsSuggestionConstraintId' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( $constraintStatusId ),
			new EntityIdValue( new ItemId( $suggestionId ) )
		);

		$this->assertThrowsConstraintParameterException(
			'parseConstraintStatusParameter',
			[ [ $constraintStatusId => [ $this->getSnakSerializer()->serialize( $snak ) ] ] ],
			'wbqc-violation-message-parameter-oneof'
		);
	}

	public function testParseConstraintStatusParameter_suggestion_enabled() {
		$constraintStatusId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintStatusId' );
		$suggestionId = self::getDefaultConfig()->get( 'WBQualityConstraintsSuggestionConstraintId' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( $constraintStatusId ),
			new EntityIdValue( new ItemId( $suggestionId ) )
		);
		$constraintParameterParser = new ConstraintParameterParser(
			new MultiConfig( [
				new HashConfig( [ 'WBQualityConstraintsEnableSuggestionConstraintStatus' => true ] ),
				self::getDefaultConfig(),
			] ),
			WikibaseRepo::getBaseDataModelDeserializerFactory(),
			'http://wikibase.example/entity/'
		);

		$parsed = $constraintParameterParser->parseConstraintStatusParameter(
			[ $constraintStatusId => [ $this->getSnakSerializer()->serialize( $snak ) ] ]
		);

		$this->assertSame( 'suggestion', $parsed );
	}

	public function testParseConstraintStatusParameter_Missing() {
		$parsed = $this->getConstraintParameterParser()->parseConstraintStatusParameter(
			[]
		);

		$this->assertNull( $parsed );
	}

	public function testParseConstraintStatusParameter_Invalid() {
		$constraintStatusId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintStatusId' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $constraintStatusId ), new EntityIdValue( new ItemId( 'Q1' ) ) );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintStatusParameter',
			[ [ $constraintStatusId => [ $this->getSnakSerializer()->serialize( $snak ) ] ] ],
			'wbqc-violation-message-parameter-oneof'
		);
	}

	public function testParseConstraintStatusParameter_UnknownError() {
		$this->assertThrowsConstraintParameterException(
			'parseExceptionParameter',
			[ [ '@error' => [] ] ],
			'wbqc-violation-message-parameters-error-unknown'
		);
	}

# endregion

# region parseSyntaxClarificationParameter
	public function testParseSyntaxClarificationParameter_SingleClarification() {
		$syntaxClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$value = new MonolingualTextValue( 'en', 'explanation' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value );

		$parsed = $this->getConstraintParameterParser()->parseSyntaxClarificationParameter(
			[ $syntaxClarificationId => [ $this->getSnakSerializer()->serialize( $snak ) ] ]
		);

		$this->assertEquals(
			new MultilingualTextValue( [ $value ] ),
			$parsed
		);
	}

	public function testParseSyntaxClarificationParameter_MultipleClarifications() {
		$syntaxClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$value1 = new MonolingualTextValue( 'en', 'explanation' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value1 );
		$value2 = new MonolingualTextValue( 'de', 'Erklärung' );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value2 );
		$value3 = new MonolingualTextValue( 'pt', 'explicação' );
		$snak3 = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value3 );

		$parsed = $this->getConstraintParameterParser()->parseSyntaxClarificationParameter(
			[ $syntaxClarificationId => [
				$this->getSnakSerializer()->serialize( $snak1 ),
				$this->getSnakSerializer()->serialize( $snak2 ),
				$this->getSnakSerializer()->serialize( $snak3 ),
			] ]
		);

		$this->assertEquals(
			new MultilingualTextValue( [ $value1, $value2, $value3 ] ),
			$parsed
		);
	}

	public function testParseSyntaxClarificationParameter_NoClarifications() {
		$parsed = $this->getConstraintParameterParser()->parseSyntaxClarificationParameter(
			[]
		);

		$this->assertEquals( new MultilingualTextValue( [] ), $parsed );
	}

	public function testParseSyntaxClarificationParameter_Invalid_MultipleValuesForLanguage() {
		$syntaxClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$value1 = new MonolingualTextValue( 'en', 'explanation' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value1 );
		$value2 = new MonolingualTextValue( 'en', 'better explanation' );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value2 );

		$this->assertThrowsConstraintParameterException(
			'parseSyntaxClarificationParameter',
			[
				[ $syntaxClarificationId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
				] ],
			],
			'wbqc-violation-message-parameter-single-per-language'
		);
	}

	public function testParseSyntaxClarificationParameter_Invalid_String() {
		$syntaxClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$value = new StringValue( 'explanation' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $syntaxClarificationId ), $value );

		$this->assertThrowsConstraintParameterException(
			'parseSyntaxClarificationParameter',
			[
				[ $syntaxClarificationId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
			],
			'wbqc-violation-message-parameter-monolingualtext'
		);
	}

	public function testParseSyntaxClarificationParameter_Invalid_Novalue() {
		$syntaxClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$snak = new PropertyNoValueSnak( new NumericPropertyId( $syntaxClarificationId ) );

		$this->assertThrowsConstraintParameterException(
			'parseSyntaxClarificationParameter',
			[
				[ $syntaxClarificationId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
			],
			'wbqc-violation-message-parameter-value'
		);
	}

# endregion

# region parseConstraintClarificationParameter
	public function testParseConstraintClarificationParameter_SingleClarification() {
		$constraintClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintClarificationId' );
		$value = new MonolingualTextValue( 'en', 'explanation' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value );

		$parsed = $this->getConstraintParameterParser()->parseConstraintClarificationParameter(
			[ $constraintClarificationId => [ $this->getSnakSerializer()->serialize( $snak ) ] ]
		);

		$this->assertEquals(
			new MultilingualTextValue( [ $value ] ),
			$parsed
		);
	}

	public function testParseConstraintClarificationParameter_MultipleClarifications() {
		$constraintClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintClarificationId' );
		$value1 = new MonolingualTextValue( 'en', 'explanation' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value1 );
		$value2 = new MonolingualTextValue( 'de', 'Erklärung' );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value2 );
		$value3 = new MonolingualTextValue( 'pt', 'explicação' );
		$snak3 = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value3 );

		$parsed = $this->getConstraintParameterParser()->parseConstraintClarificationParameter(
			[ $constraintClarificationId => [
				$this->getSnakSerializer()->serialize( $snak1 ),
				$this->getSnakSerializer()->serialize( $snak2 ),
				$this->getSnakSerializer()->serialize( $snak3 ),
			] ]
		);

		$this->assertEquals(
			new MultilingualTextValue( [ $value1, $value2, $value3 ] ),
			$parsed
		);
	}

	public function testParseConstraintClarificationParameter_NoClarifications() {
		$parsed = $this->getConstraintParameterParser()->parseConstraintClarificationParameter(
			[]
		);

		$this->assertEquals( new MultilingualTextValue( [] ), $parsed );
	}

	public function testParseConstraintClarificationParameter_Invalid_MultipleValuesForLanguage() {
		$constraintClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintClarificationId' );
		$value1 = new MonolingualTextValue( 'en', 'explanation' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value1 );
		$value2 = new MonolingualTextValue( 'en', 'better explanation' );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value2 );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintClarificationParameter',
			[
				[ $constraintClarificationId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
				] ],
			],
			'wbqc-violation-message-parameter-single-per-language'
		);
	}

	public function testParseConstraintClarificationParameter_Invalid_String() {
		$constraintClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintClarificationId' );
		$value = new StringValue( 'explanation' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $constraintClarificationId ), $value );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintClarificationParameter',
			[
				[ $constraintClarificationId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
			],
			'wbqc-violation-message-parameter-monolingualtext'
		);
	}

	public function testParseConstraintClarificationParameter_Invalid_Novalue() {
		$constraintClarificationId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintClarificationId' );
		$snak = new PropertyNoValueSnak( new NumericPropertyId( $constraintClarificationId ) );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintClarificationParameter',
			[
				[ $constraintClarificationId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
			],
			'wbqc-violation-message-parameter-value'
		);
	}

# endregion

# region parseConstraintScopeParameters
	public function testParseConstraintScopeParameters_MainSnak_ItemPropertyMediainfo() {
		$config = self::getDefaultConfig();
		$constraintScopeId = $config->get( 'WBQualityConstraintsConstraintScopeId' );
		$mainSnakId = new ItemId( $config->get( 'WBQualityConstraintsConstraintCheckedOnMainValueId' ) );
		$itemId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseItemId' ) );
		$propertyId = new ItemId( $config->get( 'WBQualityConstraintsWikibasePropertyId' ) );
		$mediainfoId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseMediaInfoId' ) );

		[ $contextTypes, $entityTypes ] = $this->getConstraintParameterParser()->parseConstraintScopeParameters(
			[ $constraintScopeId => array_map( function ( ItemId $id ) use ( $constraintScopeId ) {
				$snak = new PropertyValueSnak( new NumericPropertyId( $constraintScopeId ), new EntityIdValue( $id ) );
				return $this->getSnakSerializer()->serialize( $snak );
			}, [
				$mainSnakId,
				$itemId,
				$propertyId,
				$mediainfoId,
			] ) ],
			'Q21502838',
			Context::ALL_CONTEXT_TYPES,
			array_keys( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED )
		);

		$this->assertSame( [ Context::TYPE_STATEMENT ], $contextTypes );
		$this->assertSame( [ 'item', 'property', 'mediainfo' ], $entityTypes );
	}

	public function testParseConstraintScopeParameters_NotMainSnak_LexemeFormSense() {
		$config = self::getDefaultConfig();
		$constraintScopeId = $config->get( 'WBQualityConstraintsConstraintScopeId' );
		$qualifiersId = new ItemId( $config->get( 'WBQualityConstraintsConstraintCheckedOnQualifiersId' ) );
		$referencesId = new ItemId( $config->get( 'WBQualityConstraintsConstraintCheckedOnReferencesId' ) );
		$lexemeId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseLexemeId' ) );
		$formId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseFormId' ) );
		$senseId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseSenseId' ) );

		[ $contextTypes, $entityTypes ] = $this->getConstraintParameterParser()->parseConstraintScopeParameters(
			[ $constraintScopeId => array_map( function ( ItemId $id ) use ( $constraintScopeId ) {
				$snak = new PropertyValueSnak( new NumericPropertyId( $constraintScopeId ), new EntityIdValue( $id ) );
				return $this->getSnakSerializer()->serialize( $snak );
			}, [
				$qualifiersId,
				$referencesId,
				$lexemeId,
				$formId,
				$senseId,
			] ) ],
			'Q21502838',
			Context::ALL_CONTEXT_TYPES,
			array_keys( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED )
		);

		$this->assertSame( [ Context::TYPE_QUALIFIER, Context::TYPE_REFERENCE ], $contextTypes );
		$this->assertSame( [ 'lexeme', 'form', 'sense' ], $entityTypes );
	}

	public function testParseConstraintScopeParameters_Missing() {
		[ $contextTypes, $entityTypes ] = $this->getConstraintParameterParser()
			->parseConstraintScopeParameters( [], 'Q21502838', [], [] );

		$this->assertNull( $contextTypes );
		$this->assertNull( $entityTypes );
	}

	public function testParseConstraintScopeParameters_SeparateParameters() {
		$contextTypesParameter = 'P1';
		$entityTypesParameter = 'P2';
		$config = new MultiConfig( [
			new HashConfig( [
				'WBQualityConstraintsConstraintScopeId' => $contextTypesParameter,
				'WBQualityConstraintsConstraintEntityTypesId' => $entityTypesParameter,
			] ),
			self::getDefaultConfig(),
		] );
		$mainSnakId = new ItemId( $config->get( 'WBQualityConstraintsConstraintCheckedOnMainValueId' ) );
		$itemId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseItemId' ) );
		$constraintParameterParser = new ConstraintParameterParser(
			$config,
			WikibaseRepo::getBaseDataModelDeserializerFactory(),
			'http://wikibase.example/entity/'
		);

		[ $contextTypes, $entityTypes ] = $constraintParameterParser->parseConstraintScopeParameters(
			[
				$contextTypesParameter => [ $this->getSnakSerializer()->serialize(
					new PropertyValueSnak( new NumericPropertyId( $contextTypesParameter ), new EntityIdValue( $mainSnakId ) )
				) ],
				$entityTypesParameter => [ $this->getSnakSerializer()->serialize(
					new PropertyValueSnak( new NumericPropertyId( $entityTypesParameter ), new EntityIdValue( $itemId ) )
				) ],
			],
			'Q21502838',
			[ Context::TYPE_STATEMENT, Context::TYPE_QUALIFIER, Context::TYPE_REFERENCE ],
			array_keys( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED )
		);

		$this->assertSame( [ Context::TYPE_STATEMENT ], $contextTypes );
		$this->assertSame( [ 'item' ], $entityTypes );
	}

	public function testParseConstraintScopeParameters_InvalidContextType() {
		$config = self::getDefaultConfig();
		$constraintScopeId = $config->get( 'WBQualityConstraintsConstraintScopeId' );
		$referencesId = new ItemId( $config->get( 'WBQualityConstraintsConstraintCheckedOnReferencesId' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( $constraintScopeId ), new EntityIdValue( $referencesId ) );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintScopeParameters',
			[
				[ $constraintScopeId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
				'Q21502838',
				[ Context::TYPE_STATEMENT, Context::TYPE_QUALIFIER ],
				array_keys( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED ),
			],
			'wbqc-violation-message-invalid-scope'
		);
	}

	public function testParseConstraintScopeParameters_InvalidEntityType() {
		$config = self::getDefaultConfig();
		$constraintScopeId = $config->get( 'WBQualityConstraintsConstraintScopeId' );
		$lexemeId = new ItemId( $config->get( 'WBQualityConstraintsWikibaseLexemeId' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( $constraintScopeId ), new EntityIdValue( $lexemeId ) );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintScopeParameters',
			[
				[ $constraintScopeId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
				'Q21502838',
				[ Context::TYPE_STATEMENT, Context::TYPE_QUALIFIER, Context::TYPE_REFERENCE ],
				[ 'item', 'property' ],
			],
			'wbqc-violation-message-invalid-scope'
		);
	}

	public function testParseConstraintScopeParameter_UnknownScope() {
		$constraintScopeId = self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintScopeId' );
		$qualifiersId = new ItemId( self::getDefaultConfig()->get( 'WBQualityConstraintsConstraintCheckedOnQualifiersId' ) );
		$otherScopeId = new ItemId( 'Q1' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( $constraintScopeId ), new EntityIdValue( $qualifiersId ) );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( $constraintScopeId ), new EntityIdValue( $otherScopeId ) );

		$this->assertThrowsConstraintParameterException(
			'parseConstraintScopeParameters',
			[
				[ $constraintScopeId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
				] ],
				'Q21502838',
				Context::ALL_CONTEXT_TYPES,
				array_keys( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED ),
			],
			'wbqc-violation-message-parameter-oneof'
		);
	}

# endregion

# region parseUnitsParameter
	public function testParseUnitsParameter_NoUnitsAllowed() {
		$qualifierId = self::getDefaultConfig()->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$snak = new PropertyNoValueSnak( new NumericPropertyId( $qualifierId ) );

		$unitsParameter = $this->getConstraintParameterParser()
			->parseUnitsParameter(
				[ $qualifierId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
				'Q21514353'
			);

		$this->assertSame( [], $unitsParameter->getUnitItemIds() );
		$this->assertSame( [], $unitsParameter->getUnitQuantities() );
		$this->assertTrue( $unitsParameter->getUnitlessAllowed() );
	}

	public function testParseUnitsParameter_SomeUnitsAllowed() {
		$qualifierId = self::getDefaultConfig()->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$pid = new NumericPropertyId( $qualifierId );
		$unitId1 = new ItemId( 'Q11573' );
		$unitId2 = new ItemId( 'Q37110097' );
		$unit1 = 'http://wikibase.example/entity/Q11573';
		$unit2 = 'http://wikibase.example/entity/Q37110097';
		$snak1 = new PropertyValueSnak( $pid, new EntityIdValue( $unitId1 ) );
		$snak2 = new PropertyValueSnak( $pid, new EntityIdValue( $unitId2 ) );

		$unitsParameter = $this->getConstraintParameterParser()
			->parseUnitsParameter(
				[ $qualifierId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
				] ],
				'Q21514353'
			);

		$this->assertEquals( [ $unitId1, $unitId2 ], $unitsParameter->getUnitItemIds() );
		$unitQuantities = $unitsParameter->getUnitQuantities();
		$this->assertCount( 2, $unitQuantities );
		$this->assertSame( $unit1, $unitQuantities[0]->getUnit() );
		$this->assertSame( $unit2, $unitQuantities[1]->getUnit() );
		$this->assertFalse( $unitsParameter->getUnitlessAllowed() );
	}

	public function testParseUnitsParameter_SomeUnitsAndUnitlessAllowed() {
		$qualifierId = self::getDefaultConfig()->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$pid = new NumericPropertyId( $qualifierId );
		$unitId1 = new ItemId( 'Q11573' );
		$unitId2 = new ItemId( 'Q37110097' );
		$unit1 = 'http://wikibase.example/entity/Q11573';
		$unit2 = 'http://wikibase.example/entity/Q37110097';
		$snak1 = new PropertyValueSnak( $pid, new EntityIdValue( $unitId1 ) );
		$snak2 = new PropertyValueSnak( $pid, new EntityIdValue( $unitId2 ) );
		$snak3 = new PropertyNoValueSnak( $pid );

		$unitsParameter = $this->getConstraintParameterParser()
			->parseUnitsParameter(
				[ $qualifierId => [
					$this->getSnakSerializer()->serialize( $snak1 ),
					$this->getSnakSerializer()->serialize( $snak2 ),
					$this->getSnakSerializer()->serialize( $snak3 ),
				] ],
				'Q21514353'
			);

		$this->assertEquals( [ $unitId1, $unitId2 ], $unitsParameter->getUnitItemIds() );
		$unitQuantities = $unitsParameter->getUnitQuantities();
		$this->assertCount( 2, $unitQuantities );
		$this->assertSame( $unit1, $unitQuantities[0]->getUnit() );
		$this->assertSame( $unit2, $unitQuantities[1]->getUnit() );
		$this->assertTrue( $unitsParameter->getUnitlessAllowed() );
	}

# endregion

# region parseEntityTypesParameter
	public function testParseEntityTypesParameter_Item() {
		$qualifierId = self::getDefaultConfig()->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$itemId = new ItemId( self::getDefaultConfig()->get( 'WBQualityConstraintsWikibaseItemId' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( $qualifierId ), new EntityIdValue( $itemId ) );

		$entityTypesParameter = $this->getConstraintParameterParser()
			->parseEntityTypesParameter(
				[ $qualifierId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
				'Q52004125'
			);

		$this->assertSame( [ 'item' ], $entityTypesParameter->getEntityTypes() );
		$this->assertEquals( [ $itemId ], $entityTypesParameter->getEntityTypeItemIds() );
	}

	public function testParseEntityTypesParameter_Missing() {
		$this->assertThrowsConstraintParameterException(
			'parseEntityTypesParameter',
			[
				[],
				'Q52004125',
			],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParseEntityTypesParameter_UnknownItem() {
		$qualifierId = self::getDefaultConfig()->get( 'WBQualityConstraintsQualifierOfPropertyConstraintId' );
		$itemId = new ItemId( 'Q1' );
		$snak = new PropertyValueSnak( new NumericPropertyId( $qualifierId ), new EntityIdValue( $itemId ) );

		$this->assertThrowsConstraintParameterException(
			'parseEntityTypesParameter',
			[
				[ $qualifierId => [
					$this->getSnakSerializer()->serialize( $snak ),
				] ],
				'Q52004125',
			],
			'wbqc-violation-message-parameter-oneof'
		);
	}

# endregion

# region parseSeparatorsParameter
	public function testParseSeparatorsParameter_NoSeparators() {
		$separatorsParameter = $this->getConstraintParameterParser()
			->parseSeparatorsParameter(
				[]
			);

		$this->assertSame( [], $separatorsParameter );
	}

	public function testParseSeparatorsParameter_ThreeSeparators() {
		$separatorId = self::getDefaultConfig()->get( 'WBQualityConstraintsSeparatorId' );

		$separatorsParameter = $this->getConstraintParameterParser()
			->parseSeparatorsParameter( [ $separatorId => [
				$this->serializePropertyId( 'P1' ),
				$this->serializePropertyId( 'P2' ),
				$this->serializePropertyId( 'P4' ),
			] ]
		);

		$expected = [
			new NumericPropertyId( 'P1' ),
			new NumericPropertyId( 'P2' ),
			new NumericPropertyId( 'P4' ),
		];
		$this->assertEquals( $expected, $separatorsParameter );
	}

# endregion

# region parsePropertyScopeParameter

	/**
	 * @dataProvider provideContextTypeCombinations
	 */
	public function testParsePropertyScopeParameter( array $contextTypes ) {
		$scope = $this->getConstraintParameterParser()
			->parsePropertyScopeParameter(
				$this->propertyScopeParameter( $contextTypes ),
				'Q1'
			);

		$this->assertSame( $contextTypes, $scope );
	}

	public static function provideContextTypeCombinations() {
		return [
			[ [ Context::TYPE_STATEMENT ] ],
			[ [ Context::TYPE_QUALIFIER ] ],
			[ [ Context::TYPE_REFERENCE ] ],
			[ [ Context::TYPE_QUALIFIER, Context::TYPE_REFERENCE ] ],
		];
	}

	public function testParsePropertyScopeParameter_missing() {
		$this->assertThrowsConstraintParameterException(
			'parsePropertyScopeParameter',
			[ [], 'Q1' ],
			'wbqc-violation-message-parameter-needed'
		);
	}

	public function testParsePropertyScopeParameter_unknown() {
		$parameterId = self::getDefaultConfig()->get( 'WBQualityConstraintsPropertyScopeId' );
		$constraintParameters = [
			$parameterId => [ $this->getSnakSerializer()->serialize(
				new PropertyValueSnak(
					new NumericPropertyId( $parameterId ),
					new EntityIdValue( new ItemId( 'Q1' ) )
				)
			) ],
		];

		$this->assertThrowsConstraintParameterException(
			'parsePropertyScopeParameter',
			[ $constraintParameters, 'Q1' ],
			'wbqc-violation-message-parameter-oneof'
		);
	}

# endregion

}
