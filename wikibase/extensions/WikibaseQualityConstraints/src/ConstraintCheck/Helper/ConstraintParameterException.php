<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Exception;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;

/**
 * Exception thrown when a constraint’s parameters are invalid.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ConstraintParameterException extends Exception {

	/**
	 * @var ViolationMessage
	 */
	private $violationMessage;

	/**
	 * @param ViolationMessage $violationMessage
	 */
	public function __construct( ViolationMessage $violationMessage ) {
		$message = '⧼' . $violationMessage->getMessageKey() . '⧽';
		parent::__construct( $message );

		$this->violationMessage = $violationMessage;
	}

	/**
	 * @return ViolationMessage
	 */
	public function getViolationMessage() {
		return $this->violationMessage;
	}

}
