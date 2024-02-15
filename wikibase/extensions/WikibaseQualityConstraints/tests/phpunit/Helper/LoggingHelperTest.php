<?php

namespace WikibaseQuality\ConstraintReport\Tests\Helper;

use HashConfig;
use IBufferingStatsdDataFactory;
use Psr\Log\LoggerInterface;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Tests\DefaultConfig;
use Wikimedia\Timestamp\ConvertibleTimestamp;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class LoggingHelperTest extends \PHPUnit\Framework\TestCase {
	use DefaultConfig;

	private function getLoggingDisabledConfig() {
		return new HashConfig( [
			'WBQualityConstraintsCheckDurationInfoSeconds' => null,
			'WBQualityConstraintsCheckDurationWarningSeconds' => null,
			'WBQualityConstraintsCheckOnEntityDurationInfoSeconds' => null,
			'WBQualityConstraintsCheckOnEntityDurationWarningSeconds' => null,
		] );
	}

	/**
	 * @dataProvider provideConstraintCheckDurationsAndLogLevels
	 */
	public function testLogConstraintCheck( $durationSeconds, $expectedLevel, $expectedLimit ) {
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$constraint = new Constraint( 'test constraint id', new NumericPropertyId( 'P1' ), 'Q100', [] );
		$entity = NewItem::withId( 'Q1' )->build();
		$context = new MainSnakContext( $entity, $statement );
		$checkResult = new CheckResult(
			$context,
			$constraint,
			CheckResult::STATUS_VIOLATION,
			new ViolationMessage( 'wbqc-violation-message-single-value' )
		);

		$dataFactory = $this->createMock( IBufferingStatsdDataFactory::class );
		$dataFactory->expects( $this->once() )
			->method( 'timing' )
			->with(
				$this->identicalTo( 'wikibase.quality.constraints.check.timing.Q100-TestChecker' ),
				$this->identicalTo( $durationSeconds * 1000 )
			);

		$logger = $this->createMock( LoggerInterface::class );
		$logger->expects( $expectedLevel !== null ? $this->once() : $this->never() )
			->method( 'log' )
			->with(
				$this->identicalTo( $expectedLevel ),
				$this->identicalTo(
					'Constraint check with {constraintCheckerClassShortName} ' .
					'took longer than {limitSeconds} second(s) ' .
					'(duration: {durationSeconds} seconds).'
				),
				$this->equalTo(
					[
						'method' => __METHOD__,
						'loggingMethod' => LoggingHelper::class . '::logConstraintCheck',
						'durationSeconds' => $durationSeconds,
						'limitSeconds' => $expectedLimit,
						'constraintId' => 'test constraint id',
						'constraintPropertyId' => 'P1',
						'constraintTypeItemId' => 'Q100',
						'constraintParameters' => json_encode( [] ),
						'constraintCheckerClass' => '\Test\Namespace\TestChecker',
						'constraintCheckerClassShortName' => 'TestChecker',
						'entityId' => 'Q1',
						'statementGuid' => $statement->getGuid(),
						'resultStatus' => CheckResult::STATUS_VIOLATION,
						'resultMessage' => 'wbqc-violation-message-single-value',
					]
				)
			);

		$loggingHelper = new LoggingHelper( $dataFactory, $logger, self::getDefaultConfig() );

		$loggingHelper->logConstraintCheck(
			$context, $constraint,
			$checkResult,
			'\Test\Namespace\TestChecker', $durationSeconds,
			__METHOD__
		);
	}

	public static function provideConstraintCheckDurationsAndLogLevels() {
		return [
			'short constraint check, nothing to log' => [ 0.5, null, null ],
			'long but not extremely long constraint check, log as info' => [ 5.0, 'info', 1.0 ],
			'extremely long constraint check, log as warning' => [ 50.0, 'warning', 10.0 ],
		];
	}

	public function testLogConstraintCheckDisabled() {
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$constraint = new Constraint( 'test constraint id', new NumericPropertyId( 'P1' ), 'Q100', [] );
		$entity = NewItem::withId( 'Q1' )->build();
		$context = new MainSnakContext( $entity, $statement );
		$checkResult = new CheckResult(
			$context,
			$constraint,
			CheckResult::STATUS_VIOLATION,
			new ViolationMessage( 'wbqc-violation-message-single-value' )
		);

		$dataFactory = $this->createMock( IBufferingStatsdDataFactory::class );
		$dataFactory->expects( $this->once() )
			->method( 'timing' )
			->with(
				$this->identicalTo( 'wikibase.quality.constraints.check.timing.Q100-TestChecker' ),
				$this->identicalTo( 5000.0 )
			);

		$logger = $this->createMock( LoggerInterface::class );
		$logger->expects( $this->never() )->method( 'log' );

		$loggingHelper = new LoggingHelper( $dataFactory, $logger, $this->getLoggingDisabledConfig() );

		$loggingHelper->logConstraintCheck(
			$context, $constraint,
			$checkResult,
			'\Test\Namespace\TestChecker', 5.0,
			__METHOD__
		);
	}

	/**
	 * @dataProvider provideConstraintCheckDurationsAndLogLevelsOnEntity
	 */
	public function testLogConstraintCheckOnEntity( $durationSeconds, $expectedLevel, $expectedLimit ) {
		$entityId = new ItemId( 'Q1' );

		$dataFactory = $this->createMock( IBufferingStatsdDataFactory::class );
		$dataFactory->expects( $this->once() )
			->method( 'timing' )
			->with(
				$this->identicalTo( 'wikibase.quality.constraints.check.entity.timing' ),
				$this->identicalTo( $durationSeconds * 1000 )
			);

		$logger = $this->createMock( LoggerInterface::class );
		$logger->expects( $expectedLevel !== null ? $this->once() : $this->never() )
			->method( 'log' )
			->with(
				$this->identicalTo( $expectedLevel ),
				$this->identicalTo(
					'Full constraint check on {entityId} ' .
					'took longer than {limitSeconds} second(s) ' .
					'(duration: {durationSeconds} seconds).'
				),
				$this->equalTo(
					[
						'method' => __METHOD__,
						'loggingMethod' => LoggingHelper::class . '::logConstraintCheckOnEntity',
						'durationSeconds' => $durationSeconds,
						'limitSeconds' => $expectedLimit,
						'entityId' => 'Q1',
					]
				)
			);

		$loggingHelper = new LoggingHelper( $dataFactory, $logger, self::getDefaultConfig() );

		$loggingHelper->logConstraintCheckOnEntity(
			$entityId,
			[],
			$durationSeconds,
			__METHOD__
		);
	}

	public static function provideConstraintCheckDurationsAndLogLevelsOnEntity() {
		return [
			'short constraint check, nothing to log' => [ 5.0, null, null ],
			'long but not extremely long constraint check, log as info' => [ 10.0, 'info', 5.0 ],
			'extremely long constraint check, log as warning' => [ 120.0, 'warning', 55.0 ],
		];
	}

	public function testLogConstraintCheckOnEntityDisabled() {
		$entityId = new ItemId( 'Q1' );

		$dataFactory = $this->createMock( IBufferingStatsdDataFactory::class );
		$dataFactory->expects( $this->once() )
			->method( 'timing' )
			->with(
				$this->identicalTo( 'wikibase.quality.constraints.check.entity.timing' ),
				$this->identicalTo( 10000 )
			);

		$logger = $this->createMock( LoggerInterface::class );
		$logger->expects( $this->never() )
			->method( 'log' );

		$loggingHelper = new LoggingHelper( $dataFactory, $logger, $this->getLoggingDisabledConfig() );

		$loggingHelper->logConstraintCheckOnEntity(
			$entityId,
			[],
			10,
			__METHOD__
		);
	}

	public function testlogSparqlHelperMadeTooManyRequestsRetryAfterPresent_CallsNotice() {
		$dataFactory = $this->createMock( IBufferingStatsdDataFactory::class );
		$logger = $this->createMock( LoggerInterface::class );
		$config = $this->createMock( \Config::class );
		$timestamp = $this->createMock( ConvertibleTimestamp::class );
		$request = $this->createMock( \MWHttpRequest::class );

		$logger->expects( $this->once() )
			->method( 'notice' );

		$loggingHelper = new LoggingHelper( $dataFactory, $logger, $config );
		$loggingHelper->logSparqlHelperTooManyRequestsRetryAfterPresent( $timestamp, $request );
	}

	public function testlogSparqlHelperMadeTooManyRequestsRetryAfterMissing_CallsWarning() {
		$dataFactory = $this->createMock( IBufferingStatsdDataFactory::class );
		$logger = $this->createMock( LoggerInterface::class );
		$config = $this->createMock( \Config::class );
		$timestamp = $this->createMock( ConvertibleTimestamp::class );
		$request = $this->createMock( \MWHttpRequest::class );

		$logger->expects( $this->once() )
			->method( 'warning' );

		$loggingHelper = new LoggingHelper( $dataFactory, $logger, $config );
		$loggingHelper->logSparqlHelperTooManyRequestsRetryAfterInvalid( $request );
	}

}
