<?php

namespace WikibaseQuality\ConstraintReport\Tests\Fake;

use Wikimedia\Timestamp\ConvertibleTimestamp;

/**
 * Always returns the invalid timestamp string 'invalid timestamp'.
 *
 * @license GPL-2.0-or-later
 */
class InvalidConvertibleTimestamp extends ConvertibleTimestamp {

	public function __construct() {
	}

	public function getTimestamp( $style = TS_UNIX ) {
		return 'invalid timestamp';
	}

}
