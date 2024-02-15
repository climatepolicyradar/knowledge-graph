<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Result;

use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use DataValues\StringValue;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Tests\Fake\AppendingContextCursor;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult
 *
 * @group WikibaseQualityConstraints
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class CheckResultTest extends \MediaWikiUnitTestCase {

	public function testConstructAndGetters_Context() {
		$propertyId = new NumericPropertyId( 'P1' );
		$entityId = new ItemId( 'Q1' );
		$snak = new PropertyValueSnak( $propertyId, new StringValue( 'Foo' ) );
		$constraintId = '1';
		$constraint = new Constraint( $constraintId, new NumericPropertyId( 'P1' ), 'Q100', [] );
		$status = CheckResult::STATUS_COMPLIANCE;
		$message = new ViolationMessage( 'wbqc-violation-message-single-value' );
		$context = new FakeSnakContext( $snak, new Item( $entityId ) );
		$metadata = Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 42 ) );

		$checkResult = new CheckResult( $context, $constraint, $status, $message );
		$checkResult->withMetadata( $metadata );

		$this->assertEquals( $context->getCursor(), $checkResult->getContextCursor() );
		$this->assertSame( $snak->getType(), $checkResult->getSnakType() );
		$this->assertSame( $snak->getDataValue(), $checkResult->getDataValue() );
		$this->assertSame( $constraint, $checkResult->getConstraint() );
		$this->assertSame( $constraintId, $checkResult->getConstraintId() );
		$this->assertSame( $status, $checkResult->getStatus() );
		$this->assertSame( $message, $checkResult->getMessage() );
		$this->assertSame( $metadata, $checkResult->getMetadata() );
	}

	public function testConstructAndGetters_Cursor() {
		$propertyId = new NumericPropertyId( 'P1' );
		$entityId = new ItemId( 'Q1' );
		$snak = new PropertyValueSnak( $propertyId, new StringValue( 'Foo' ) );
		$constraintId = '1';
		$constraint = new Constraint( $constraintId, new NumericPropertyId( 'P1' ), 'Q100', [] );
		$status = CheckResult::STATUS_COMPLIANCE;
		$message = new ViolationMessage( 'wbqc-violation-message-single-value' );
		$context = new AppendingContextCursor();
		$metadata = Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 42 ) );

		$checkResult = new CheckResult( $context, $constraint, $status, $message );
		$checkResult->withMetadata( $metadata );

		$this->assertSame( $context, $checkResult->getContextCursor() );
		$this->assertNull( $checkResult->getSnakType() );
		$this->assertNull( $checkResult->getDataValue() );
		$this->assertSame( $constraint, $checkResult->getConstraint() );
		$this->assertSame( $constraintId, $checkResult->getConstraintId() );
		$this->assertSame( $status, $checkResult->getStatus() );
		$this->assertSame( $message, $checkResult->getMessage() );
		$this->assertSame( $metadata, $checkResult->getMetadata() );
	}

	public function testSetStatus() {
		$context = new FakeSnakContext( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) );
		$constraint = new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] );
		$checkResult = new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION );

		$this->assertSame( CheckResult::STATUS_VIOLATION, $checkResult->getStatus() );

		$checkResult->setStatus( CheckResult::STATUS_WARNING );
		$this->assertSame( CheckResult::STATUS_WARNING, $checkResult->getStatus() );
	}

	public function testSetMessage(): void {
		$context = new FakeSnakContext( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) );
		$constraint = new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] );
		$checkResult = new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION );

		$this->assertNull( $checkResult->getMessage() );

		$message = new ViolationMessage( 'wbqc-violation-message-single-value' );
		$checkResult->setMessage( $message );
		$this->assertSame( $message, $checkResult->getMessage() );
	}

	public function testSetConstraintClarification(): void {
		$context = new FakeSnakContext( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) );
		$constraint = new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] );
		$checkResult = new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION );

		$this->assertSame( [], $checkResult->getConstraintClarification()->getArrayValue() );

		$constraintClarification = new MultilingualTextValue( [
			new MonolingualTextValue( 'en', 'constraint clarification' ),
		] );
		$checkResult->setConstraintClarification( $constraintClarification );
		$this->assertSame( $constraintClarification, $checkResult->getConstraintClarification() );
	}

}
