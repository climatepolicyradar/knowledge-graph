<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use DataValues\TimeValue;
use LogicException;

/**
 * A TimeValue describing the current point in time in the Proleptic Gregorian calendar.
 * The value is always given in UTC.
 *
 * Note that this is not a full-featured TimeValue.
 * It cannot be serialized and does not support any operations
 * other than {@link getTime} and {@link equals}.
 *
 * @license GPL-2.0-or-later
 * @phan-file-suppress PhanPluginNeverReturnMethod
 */
class NowValue extends TimeValue {

	public function __construct() {
		// missing parent::construct() is deliberate – this is not a full-featured TimeValue
	}

	/**
	 * The current time at the time when the function is called.
	 *
	 * @return string
	 */
	public function getTime() {
		return gmdate( '+Y-m-d\TH:i:s\Z' );
	}

	public function getTimezone() {
		return 0;
	}

	public function getCalendarModel() {
		return parent::CALENDAR_GREGORIAN;
	}

	public function getArrayValue() {
		throw new LogicException( 'NowValue should never be serialized' );
	}

	public function equals( $value ) {
		return get_class( $value ) === self::class;
	}

}
