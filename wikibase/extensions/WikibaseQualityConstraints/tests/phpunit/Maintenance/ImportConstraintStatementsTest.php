<?php

namespace WikibaseQuality\ConstraintReport\Tests\Maintenance;

use HashConfig;
use MediaWiki\Tests\Maintenance\MaintenanceBaseTestCase;
use Wikibase\Lib\Tests\Store\MockPropertyInfoLookup;
use WikibaseQuality\ConstraintReport\Job\UpdateConstraintsTableJob;
use WikibaseQuality\ConstraintReport\Maintenance\ImportConstraintStatements;
use Wikimedia\Rdbms\ILBFactory;

/**
 * @covers \WikibaseQuality\ConstraintReport\Maintenance\ImportConstraintStatements
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ImportConstraintStatementsTest extends MaintenanceBaseTestCase {

	public function getMaintenanceClass() {
		return ImportConstraintStatements::class;
	}

	public function testConstraintStatementsDisabled() {
		$this->maintenance->setConfig( new HashConfig( [
			'WBQualityConstraintsEnableConstraintsImportFromStatements' => false,
		] ) );

		$this->maintenance->execute();

		$this->expectOutputString( 'Constraint statements are not enabled. Aborting.' );
	}

	public function testNoProperties() {
		$this->maintenance->setupServices = function () {
			$this->maintenance->propertyInfoLookup = new MockPropertyInfoLookup( [] );
			$this->maintenance->lbFactory = $this->createMock( ILBFactory::class );
		};
		$this->maintenance->newUpdateConstraintsTableJob = function () {
			$this->fail( 'newUpdateConstraintsTableJob should not be called' );
		};

		$this->maintenance->execute();

		$this->expectOutputString( '' );
	}

	public function testTwoProperties() {
		$this->maintenance->setupServices = function () {
			$this->maintenance->propertyInfoLookup = new MockPropertyInfoLookup( [
				'P1' => [],
				'P3' => [],
			] );
			$this->maintenance->lbFactory = $this->createMock( ILBFactory::class );
			$this->maintenance->lbFactory->expects( $this->once() )
				->method( 'waitForReplication' );
		};

		$call = 0;
		$this->maintenance->newUpdateConstraintsTableJob = function ( $propertyIdSerialization ) use ( &$call ) {
			$mock = $this->createMock( UpdateConstraintsTableJob::class );
			$mock->expects( $this->once() )
				->method( 'run' )
				->with();
			switch ( ++$call ) {
				case 1:
					$this->assertSame( 'P1', $propertyIdSerialization );
					return $mock;
				case 2:
					$this->assertSame( 'P3', $propertyIdSerialization );
					return $mock;
				default:
					$this->fail( 'newUpdateConstraintsTableJob should only be called twice' );
					return $mock; // unreachable but just in case
			}
		};

		$this->maintenance->execute();

		$this->expectOutputRegex( '/^' .
			'Importing constraint statements for +P1... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P3... done in +\d+\.\d+ ms.\n' .
			'Waiting for replication... done in +\d+\.\d+ ms.\n' .
			'$/' );
		$this->assertSame( 2, $call, 'newUpdateConstraintsTableJob should have been called twice' );
	}

	public function testTenPropertiesBatchSizeFive() {
		$this->maintenance->setupServices = function () {
			for ( $i = 1; $i <= 10; $i++ ) {
				$id = "P$i";
				$propertyInfos[$id] = [];
			}
			$this->maintenance->propertyInfoLookup = new MockPropertyInfoLookup( $propertyInfos );
			$this->maintenance->lbFactory = $this->createMock( ILBFactory::class );
			$this->maintenance->lbFactory->expects( $this->exactly( 2 ) )
				->method( 'waitForReplication' );
		};
		$call = 0;
		$this->maintenance->newUpdateConstraintsTableJob = function ( $_ ) use ( &$call ) {
			$call++;
			return $this->createMock( UpdateConstraintsTableJob::class );
		};
		$this->maintenance->loadWithArgv( [ '--batch-size=5' ] );

		$this->maintenance->execute();

		$this->expectOutputRegex( '/^' .
			'Importing constraint statements for +P1... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P2... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P3... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P4... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P5... done in +\d+\.\d+ ms.\n' .
			'Waiting for replication... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P6... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P7... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P8... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P9... done in +\d+\.\d+ ms.\n' .
			'Importing constraint statements for +P10... done in +\d+\.\d+ ms.\n' .
			'Waiting for replication... done in +\d+\.\d+ ms.\n' .
			'$/' );
		$this->assertSame( 10, $call,
			'newUpdateConstraintsTableJob should have been called 10 times' );
	}

	public function testDefaultNewUpdateConstraintsTableJob() {
		$job = call_user_func( $this->maintenance->newUpdateConstraintsTableJob, 'P1234' );

		$this->assertInstanceOf( UpdateConstraintsTableJob::class, $job );
		$params = $job->getParams();
		$this->assertArrayHasKey( 'propertyId', $params );
		$this->assertSame( 'P1234', $params['propertyId'] );
	}

}
