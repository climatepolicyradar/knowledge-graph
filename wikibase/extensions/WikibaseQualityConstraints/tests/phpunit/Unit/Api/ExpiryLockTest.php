<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Api;

use HashBagOStuff;
use WikibaseQuality\ConstraintReport\Api\ExpiryLock;
use WikibaseQuality\ConstraintReport\Tests\Fake\InvalidConvertibleTimestamp;
use Wikimedia\Timestamp\ConvertibleTimestamp;

/**
 * @covers \WikibaseQuality\ConstraintReport\Api\ExpiryLock
 *
 * @group WikibaseQualityConstraints
 *
 * @author Matthias Geisler
 * @license GPL-2.0-or-later
 */
class ExpiryLockTest extends \MediaWikiUnitTestCase {

	public function tearDown(): void {
		ConvertibleTimestamp::setFakeTime( false );
		parent::tearDown();
	}

	public function testLock_whenNotLockedYet_returnsTrue() {
		$lock = new ExpiryLock( new HashBagOStuff() );

		$this->assertTrue( $lock->lock( 'fooId', new ConvertibleTimestamp( '10' ) ) );
	}

	public function testLock_whenAlreadyLocked_returnsFalse() {
		$lock = new ExpiryLock( new HashBagOStuff() );
		$id = 'fooId';
		$expiryTimeStamp = new ConvertibleTimestamp( '10' );
		ConvertibleTimestamp::setFakeTime( '9' );

		$lock->lock( $id, $expiryTimeStamp );

		$this->assertFalse( $lock->lock( $id, $expiryTimeStamp ) );
	}

	public function testIsLocked_whenInvalidTimestampStored_returnsFalse() {
		$lock = new ExpiryLock( new HashBagOStuff() );
		$id = 'fooId';
		$invalidTimestamp = new InvalidConvertibleTimestamp();

		$lock->lock( $id, $invalidTimestamp );

		$this->assertFalse( $lock->isLocked( 'fooId' ) );
	}

	public function testIsLocked_whenNotLockedYet_returnsFalse() {
		$lock = new ExpiryLock( new HashBagOStuff() );

		$this->assertFalse( $lock->isLocked( 'fooId' ) );
	}

	public function testIsLocked_whenLockedButExpired_returnsFalse() {
		$lock = new ExpiryLock( new HashBagOStuff() );
		$id = 'fooId';
		ConvertibleTimestamp::setFakeTime( '11' );

		$lock->lock( $id, new ConvertibleTimestamp( '10' ) );

		$this->assertFalse( $lock->isLocked( $id ) );
	}

	public function testIsLocked_whenLockedAndNotExpiredYet_returnsTrue() {
		$lock = new ExpiryLock( new HashBagOStuff() );
		$id = 'fooId';
		ConvertibleTimestamp::setFakeTime( '40' );

		$lock->lock( $id, new ConvertibleTimestamp( '50' ) );

		$this->assertTrue( $lock->isLocked( $id ) );
	}

}
