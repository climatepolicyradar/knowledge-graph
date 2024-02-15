<?php

namespace WikibaseQuality\ConstraintReport\Api;

use BagOStuff;
use Wikimedia\Assert\Assert;
use Wikimedia\Assert\ParameterTypeException;
use Wikimedia\Timestamp\ConvertibleTimestamp;
use Wikimedia\Timestamp\TimestampException;

/**
 * A thin wrapper around a BagOStuff
 * that caches a timestamp and create a lock
 * until a certain time is over.
 *
 * @author Matthias Geisler
 * @license GPL-2.0-or-later
 */
class ExpiryLock {

	/**
	 * @var BagOStuff
	 */
	private $cache;

	/**
	 * @param BagOStuff $cache
	 */
	public function __construct( BagOStuff $cache ) {
		$this->cache = $cache;
	}

	/**
	 * @param string $id of the lock
	 *
	 * @return string cache key
	 *
	 * @throws \Wikimedia\Assert\ParameterTypeException
	 */
	private function makeKey( $id ) {
		if ( empty( trim( $id ) ) ) {
			throw new ParameterTypeException( '$id', 'non-empty string' );
		}

		Assert::parameterType( 'string', $id, '$id' );

		return $this->cache->makeKey(
			'WikibaseQualityConstraints',
			'ExpiryLock',
			(string)$id
		);
	}

	/**
	 * @param string $id of the lock
	 * @param ConvertibleTimestamp $expiryTimestamp
	 *
	 * @return boolean success
	 *
	 * @throws \Wikimedia\Assert\ParameterTypeException
	 */
	public function lock( $id, ConvertibleTimestamp $expiryTimestamp ) {

		$cacheId = $this->makeKey( $id );

		if ( !$this->isLockedInternal( $cacheId ) ) {
			return $this->cache->set(
				$cacheId,
				$expiryTimestamp->getTimestamp( TS_UNIX ),
				(int)$expiryTimestamp->getTimestamp( TS_UNIX )
			);
		} else {
			return false;
		}
	}

	/**
	 * @param string $cacheId the converted cache id
	 *
	 * @return boolean representing if the Lock is Locked
	 *
	 * @throws \Wikimedia\Assert\ParameterTypeException
	 */
	private function isLockedInternal( $cacheId ) {
		$expiryTime = $this->cache->get( $cacheId );
		if ( !$expiryTime ) {
			return false;
		}

		try {
			$lockExpiryTimeStamp = new ConvertibleTimestamp( $expiryTime );
		} catch ( TimestampException $exception ) {
			return false;
		}

		$now = new ConvertibleTimestamp();
		if ( $now->timestamp < $lockExpiryTimeStamp->timestamp ) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * @param string $id of the lock
	 *
	 * @return boolean representing if the Lock is Locked
	 *
	 * @throws \Wikimedia\Assert\ParameterTypeException
	 */
	public function isLocked( $id ) {
		return $this->isLockedInternal( $this->makeKey( $id ) );
	}

}
