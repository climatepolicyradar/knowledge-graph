<?php

namespace WikibaseQuality\ConstraintReport;

/**
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ConstraintSerializer {

	/**
	 * @var bool
	 */
	private $serializeConstraintParameters;

	/**
	 * @param bool $serializeConstraintParameters Whether to serialize constraint parameters or not.
	 */
	public function __construct( $serializeConstraintParameters = true ) {
		$this->serializeConstraintParameters = $serializeConstraintParameters;
	}

	/**
	 * @param Constraint $constraint
	 * @return array
	 */
	public function serialize( Constraint $constraint ) {
		$serialization = [
			'id' => $constraint->getConstraintId(),
			'pid' => $constraint->getPropertyId()->getSerialization(),
			'qid' => $constraint->getConstraintTypeItemId(),
		];
		if ( $this->serializeConstraintParameters ) {
			$serialization['params'] = $constraint->getConstraintParameters();
		}
		return $serialization;
	}

}
