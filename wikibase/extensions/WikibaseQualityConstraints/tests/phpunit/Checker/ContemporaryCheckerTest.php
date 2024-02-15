<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use DataValues\TimeValue;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ContemporaryChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\RangeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Tests\DefaultConfig;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers \WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ContemporaryChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author David Abián
 * @license GPL-2.0-or-later
 */
class ContemporaryCheckerTest extends \PHPUnit\Framework\TestCase {

	use DefaultConfig;
	use ResultAssertions;

	/**
	 * @var RangeCheckerHelper
	 */
	private $rangeCheckerHelper;

	/**
	 * @var mixed Array of arbitrary timestamps ordered chronologically.
	 */
	private $timestamps;

	/**
	 * @var string Arbitrary ID of the property that will link subject and object.
	 */
	private $linkingPropertyId;

	/**
	 * @var array IDs of the properties that state the start time value of the entities.
	 */
	private $startPropertyIds;

	/**
	 * @var array IDs of the properties that state the end time value of the entities.
	 */
	private $endPropertyIds;

	/**
	 * @throws \ConfigException
	 */
	protected function setUp(): void {
		parent::setUp();
		$this->startPropertyIds = self::getDefaultConfig()
			->get( ContemporaryChecker::CONFIG_VARIABLE_START_PROPERTY_IDS );
		$this->endPropertyIds = self::getDefaultConfig()
			->get( ContemporaryChecker::CONFIG_VARIABLE_END_PROPERTY_IDS );
		$this->rangeCheckerHelper = new RangeCheckerHelper( self::getDefaultConfig() );
		$this->linkingPropertyId = 'P123456';
		$this->timestamps = [
			'-000401862-05-31T00:00:00Z', // BCE
			'+000000015-04-30T00:00:00Z', // CE
			'+000000974-03-10T00:00:00Z', // CE
			'+987654321-01-08T00:00:00Z', // CE
		];
	}

	/**
	 * @dataProvider provideStandardStatements
	 *
	 * @param InMemoryEntityLookup $lookup
	 * @param Item $subjectItem
	 * @param Statement $statement
	 * @param string $expectedStatus
	 *
	 * @throws \ConfigException
	 */
	public function testContemporaryConstraintStandardStatements(
		InMemoryEntityLookup $lookup,
		Item $subjectItem,
		Statement $statement,
		$expectedStatus
	) {
		$constraint = $this->getConstraintMock();
		$checker = new ContemporaryChecker( $lookup, $this->rangeCheckerHelper, self::getDefaultConfig() );
		$checkResult = $checker->checkConstraint( new MainSnakContext( $subjectItem, $statement ), $constraint );
		if ( $expectedStatus === CheckResult::STATUS_COMPLIANCE ) {
			$this->assertCompliance( $checkResult );
		} elseif ( $expectedStatus === CheckResult::STATUS_VIOLATION ) {
			$this->assertViolation( $checkResult );
		} else {
			throw new \InvalidArgumentException(
				'$expectedStatus should be STATUS_COMPLIANCE or STATUS_VIOLATION. '
				. 'Provided: ' . $expectedStatus . '.'
			);
		}
	}

	/**
	 * @throws \ConfigException
	 */
	public static function provideStandardStatements(): iterable {
		$orderedTimestamps = [
			'-000001862-05-31T00:00:00Z', // BCE
			'+000000015-04-30T00:00:00Z', // CE
			'+000000974-03-10T00:00:00Z', // CE
			'+987654321-01-08T00:00:00Z', // CE
		];

		/**
		 * Array whose arrays have:
		 *  [0] Start time for the subject (index for $orderedTimestamps)
		 *  [1] End time for the subject (index for $orderedTimestamps)
		 *  [2] Start time for the value (index for $orderedTimestamps)
		 *  [3] End time for the value (index for $orderedTimestamps)
		 *  [4] Expected result (compliance or violation)
		 *
		 * The indexes (elements 0, 1, 2, 3) can be interpreted as abstract timestamps
		 * because the array $orderedTimestamps is ordered chronologically.
		 */
		$violationMatrix = [
			[ 0, 0, 0, 0, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 0, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 0, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 0, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 0, 1, 1, CheckResult::STATUS_VIOLATION ],
			[ 0, 0, 1, 2, CheckResult::STATUS_VIOLATION ],
			[ 0, 0, 1, 3, CheckResult::STATUS_VIOLATION ],
			[ 0, 0, 2, 2, CheckResult::STATUS_VIOLATION ],
			[ 0, 0, 2, 3, CheckResult::STATUS_VIOLATION ],
			[ 0, 0, 3, 3, CheckResult::STATUS_VIOLATION ],
			[ 0, 1, 0, 0, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 1, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 1, 2, 2, CheckResult::STATUS_VIOLATION ],
			[ 0, 1, 2, 3, CheckResult::STATUS_VIOLATION ],
			[ 0, 1, 3, 3, CheckResult::STATUS_VIOLATION ],
			[ 0, 2, 0, 0, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 1, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 2, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 2, 3, 3, CheckResult::STATUS_VIOLATION ],
			[ 0, 3, 0, 0, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 1, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 2, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 0, 3, 3, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 0, 0, CheckResult::STATUS_VIOLATION ],
			[ 1, 1, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 1, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 1, 2, 2, CheckResult::STATUS_VIOLATION ],
			[ 1, 1, 2, 3, CheckResult::STATUS_VIOLATION ],
			[ 1, 1, 3, 3, CheckResult::STATUS_VIOLATION ],
			[ 1, 2, 0, 0, CheckResult::STATUS_VIOLATION ],
			[ 1, 2, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 1, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 2, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 2, 3, 3, CheckResult::STATUS_VIOLATION ],
			[ 1, 3, 0, 0, CheckResult::STATUS_VIOLATION ],
			[ 1, 3, 0, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 1, 1, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 2, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 1, 3, 3, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 0, 0, CheckResult::STATUS_VIOLATION ],
			[ 2, 2, 0, 1, CheckResult::STATUS_VIOLATION ],
			[ 2, 2, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 1, 1, CheckResult::STATUS_VIOLATION ],
			[ 2, 2, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 2, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 2, 3, 3, CheckResult::STATUS_VIOLATION ],
			[ 2, 3, 0, 0, CheckResult::STATUS_VIOLATION ],
			[ 2, 3, 0, 1, CheckResult::STATUS_VIOLATION ],
			[ 2, 3, 0, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 3, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 3, 1, 1, CheckResult::STATUS_VIOLATION ],
			[ 2, 3, 1, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 3, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 3, 2, 2, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 3, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 2, 3, 3, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 3, 3, 0, 0, CheckResult::STATUS_VIOLATION ],
			[ 3, 3, 0, 1, CheckResult::STATUS_VIOLATION ],
			[ 3, 3, 0, 2, CheckResult::STATUS_VIOLATION ],
			[ 3, 3, 0, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 3, 3, 1, 1, CheckResult::STATUS_VIOLATION ],
			[ 3, 3, 1, 2, CheckResult::STATUS_VIOLATION ],
			[ 3, 3, 1, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 3, 3, 2, 2, CheckResult::STATUS_VIOLATION ],
			[ 3, 3, 2, 3, CheckResult::STATUS_COMPLIANCE ],
			[ 3, 3, 3, 3, CheckResult::STATUS_COMPLIANCE ],
		];

		$currentItemId = 1;
		$lookup = new InMemoryEntityLookup();
		$toBeReturned = [];
		foreach ( $violationMatrix as $violationArray ) {
			$generatedLinkedPair = self::generateLinkedItemPair(
				$lookup,
				'P1',
				'Q' . $currentItemId++,
				'Q' . $currentItemId++,
				$orderedTimestamps[$violationArray[0]],
				$orderedTimestamps[$violationArray[1]],
				$orderedTimestamps[$violationArray[2]],
				$orderedTimestamps[$violationArray[3]]
			);
			$lookup = $generatedLinkedPair[0];
			$subjectItem = $generatedLinkedPair[1];
			$statement = $generatedLinkedPair[2];
			array_push( $toBeReturned, [ $lookup, $subjectItem, $statement, $violationArray[4] ] );
		}
		return $toBeReturned;
	}

	public function testContemporaryConstraintEternalSubject() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintEternalObject() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintEternalEntities() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintDeletedObject() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q255' )->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintNoObject() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( NewStatement::noValueFor( new NumericPropertyId( $this->linkingPropertyId ) ) )
			->build();
		$this->saveAndCheck( $subjectItem, null, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSomeObject() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( NewStatement::someValueFor( new NumericPropertyId( $this->linkingPropertyId ) ) )
			->build();
		$this->saveAndCheck( $subjectItem, null, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintUndefinedSubjectStartCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintUndefinedSubjectStartViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintUndefinedSubjectEndCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintUndefinedSubjectEndViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintUndefinedObjectStartCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintUndefinedObjectStartViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintUndefinedObjectEndCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintUndefinedObjectEndViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintNoSubjectStartCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( NewStatement::noValueFor( $this->startPropertyIds[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintNoSubjectStartViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( NewStatement::noValueFor( $this->startPropertyIds[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintNoSubjectEndCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( NewStatement::noValueFor( $this->endPropertyIds[0] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintNoSubjectEndViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( NewStatement::noValueFor( $this->endPropertyIds[0] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintNoObjectStartCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( NewStatement::noValueFor( $this->startPropertyIds[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintNoObjectStartViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( NewStatement::noValueFor( $this->startPropertyIds[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintNoObjectEndCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( NewStatement::noValueFor( $this->endPropertyIds[0] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintNoObjectEndViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( NewStatement::noValueFor( $this->endPropertyIds[0] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSomeSubjectStart() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( NewStatement::someValueFor( $this->startPropertyIds[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSomeSubjectEnd() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( NewStatement::someValueFor( $this->endPropertyIds[0] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSomeObjectStart() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( NewStatement::someValueFor( $this->startPropertyIds[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSomeObjectEnd() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement(
				NewStatement::someValueFor( $this->endPropertyIds[0] )
			)
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSubjectEndBeforeStart1() {
		// Compliance expected if start and end times were exchanged
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSubjectEndBeforeStart2() {
		// Violation expected if start and end times were exchanged
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintObjectEndBeforeStart1() {
		// Compliance expected if start and end times were exchanged
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[0] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintObjectEndBeforeStart2() {
		// Violation expected if start and end times were exchanged
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[0] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintDeprecatedLink1() {
		// Compliance expected if not deprecated, but STATUS_DEPRECATED expected if deprecated
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' )->withDeprecatedRank() )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_DEPRECATED );
	}

	public function testContemporaryConstraintDeprecatedLink2() {
		// Violation expected if not deprecated, but STATUS_DEPRECATED expected if deprecated
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' )->withDeprecatedRank() )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_DEPRECATED );
	}

	public function testContemporaryConstraintPreferredLinkCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' )->withPreferredRank() )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintPreferredLinkViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' )->withPreferredRank() )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSeveralSubjectStartStatementsCompliance() {
		if ( count( $this->startPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one start property ID defined, cannot test behavior for multiple start statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSeveralSubjectStartStatementsViolation() {
		if ( count( $this->startPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one start property ID defined, cannot test behavior for multiple start statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSeveralSubjectEndStatementsCompliance() {
		if ( count( $this->endPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one end property ID defined, cannot test behavior for multiple end statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[1], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSeveralSubjectEndStatementsViolation() {
		if ( count( $this->endPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one end property ID defined, cannot test behavior for multiple end statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[1], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSeveralObjectStartStatementsCompliance() {
		if ( count( $this->startPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one start property ID defined, cannot test behavior for multiple start statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[1] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSeveralObjectStartStatementsViolation() {
		if ( count( $this->startPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one start property ID defined, cannot test behavior for multiple start statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[1], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSeveralObjectEndStatementsCompliance() {
		if ( count( $this->endPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one end property ID defined, cannot test behavior for multiple end statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[1], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSeveralObjectEndStatementsViolation() {
		if ( count( $this->endPropertyIds ) <= 1 ) {
			$this->markTestSkipped(
				'Only one end property ID defined, cannot test behavior for multiple end statements.'
			);
		}
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[1], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSubjectStartDeprecatedCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement(
				self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] )->withDeprecatedRank()
			)
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSubjectStartDeprecatedViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement(
				self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] )->withDeprecatedRank()
			)
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintSubjectEndDeprecatedCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement(
				self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] )->withDeprecatedRank()
			)
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintSubjectEndDeprecatedViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement(
				self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] )->withDeprecatedRank()
			)
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintObjectStartDeprecatedCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement(
				self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] )->withDeprecatedRank()
			)
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintObjectStartDeprecatedViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement(
				self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[1] )->withDeprecatedRank()
			)
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	public function testContemporaryConstraintObjectEndDeprecatedCompliance() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[2] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement(
				self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[1] )->withDeprecatedRank()
			)
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_COMPLIANCE );
	}

	public function testContemporaryConstraintObjectEndDeprecatedViolation() {
		$subjectItem = NewItem::withId( 'Q1' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] ) )
			->andStatement( $this->newLinkingStatement( 'Q2' ) )
			->build();
		$valueItem = NewItem::withId( 'Q2' )
			->andStatement( self::newTimeStatement( $this->startPropertyIds[0], $this->timestamps[0] ) )
			->andStatement( self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[2] ) )
			->andStatement(
				self::newTimeStatement( $this->endPropertyIds[0], $this->timestamps[3] )->withDeprecatedRank()
			)
			->build();
		$this->saveAndCheck( $subjectItem, $valueItem, CheckResult::STATUS_VIOLATION );
	}

	/**
	 * @return Constraint
	 */
	private function getConstraintMock() {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			->willReturn( [] );
		$mock->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q25796498' );
		return $mock;
	}

	/**
	 * @param InMemoryEntityLookup $lookup
	 * @param string $linkingPropertyId
	 * @param string $subjectItemId
	 * @param string $valueItemId
	 * @param string $subjectStartTimestamp
	 * @param string $subjectEndTimestamp
	 * @param string $valueStartTimestamp
	 * @param string $valueEndTimestamp
	 *
	 * @return array with lookup, subject item and linking statement
	 * @throws \ConfigException
	 */
	private static function generateLinkedItemPair(
		InMemoryEntityLookup $lookup,
		$linkingPropertyId,
		$subjectItemId,
		$valueItemId,
		$subjectStartTimestamp,
		$subjectEndTimestamp,
		$valueStartTimestamp,
		$valueEndTimestamp
	) {
		$startPropertyIds = self::getDefaultConfig()
			->get( ContemporaryChecker::CONFIG_VARIABLE_START_PROPERTY_IDS );
		$endPropertyIds = self::getDefaultConfig()
			->get( ContemporaryChecker::CONFIG_VARIABLE_END_PROPERTY_IDS );
		$subjectItem = NewItem::withId( $subjectItemId )
			->andStatement( self::newTimeStatement( $startPropertyIds[0], $subjectStartTimestamp ) )
			->andStatement( self::newTimeStatement( $endPropertyIds[0], $subjectEndTimestamp ) )
			->andStatement(
				NewStatement::forProperty( $linkingPropertyId )
					->withValue( new ItemId( $valueItemId ) )
			)
			->build();
		$lookup->addEntity( $subjectItem );
		$valueItem = NewItem::withId( $valueItemId )
			->andStatement( self::newTimeStatement( $startPropertyIds[0], $valueStartTimestamp ) )
			->andStatement( self::newTimeStatement( $endPropertyIds[0], $valueEndTimestamp ) )
			->build();
		$lookup->addEntity( $valueItem );
		return [
			$lookup,
			$subjectItem,
			self::getLinkingStatement( $subjectItem, $linkingPropertyId ),
		];
	}

	/**
	 * @param Item $subjectItem
	 * @param ?Item $valueItem
	 * @param string $expectedStatus
	 *
	 * @throws \ConfigException
	 */
	private function saveAndCheck( Item $subjectItem, ?Item $valueItem, $expectedStatus ) {
		$lookup = new InMemoryEntityLookup();
		$lookup->addEntity( $subjectItem );
		if ( $valueItem != null ) {
			$lookup->addEntity( $valueItem );
		}
		$constraint = $this->getConstraintMock();
		$checker = new ContemporaryChecker( $lookup, $this->rangeCheckerHelper, self::getDefaultConfig() );
		$statement = self::getLinkingStatement( $subjectItem, $this->linkingPropertyId );
		$checkResult = $checker->checkConstraint( new MainSnakContext( $subjectItem, $statement ), $constraint );
		if ( $expectedStatus === CheckResult::STATUS_COMPLIANCE ) {
			$this->assertCompliance( $checkResult );
		} elseif ( $expectedStatus === CheckResult::STATUS_VIOLATION ) {
			$this->assertViolation( $checkResult );
		} elseif ( $expectedStatus === CheckResult::STATUS_DEPRECATED ) {
			$this->assertDeprecation( $checkResult );
		} else {
			throw new \InvalidArgumentException(
				'$expectedStatus should be STATUS_COMPLIANCE, STATUS_VIOLATION or '
				. 'STATUS_DEPRECATED. Provided: ' . $expectedStatus . '.'
			);
		}
	}

	/**
	 * @param string $extremePropertyId
	 * @param string $timestamp
	 *
	 * @return NewStatement
	 */
	private static function newTimeStatement( $extremePropertyId, $timestamp ) {
		return NewStatement::forProperty( $extremePropertyId )
			->withValue(
				new TimeValue(
					$timestamp, 0, 0, 0,
					TimeValue::PRECISION_DAY,
					TimeValue::CALENDAR_GREGORIAN
				)
			);
	}

	/**
	 * @param Item $subjectItem
	 * @param string $linkingPropertyId
	 *
	 * @return mixed
	 */
	private static function getLinkingStatement( Item $subjectItem, $linkingPropertyId ) {
		return $subjectItem
			->getStatements()
			->getByPropertyId( new NumericPropertyId( $linkingPropertyId ) )
			->toArray()[0];
	}

	/**
	 * @param string $valueItemId
	 *
	 * @return NewStatement
	 */
	private function newLinkingStatement( $valueItemId ) {
		return NewStatement::forProperty( $this->linkingPropertyId )
			->withValue( new ItemId( $valueItemId ) );
	}

}
