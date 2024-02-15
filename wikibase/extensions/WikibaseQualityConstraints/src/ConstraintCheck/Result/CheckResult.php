<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Result;

use DataValues\DataValue;
use DataValues\MultilingualTextValue;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;

/**
 * Used for getting information about the result of a constraint check
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class CheckResult {

	// Constants for statuses
	/**
	 * The statement satisfies the constraint.
	 */
	public const STATUS_COMPLIANCE = 'compliance';
	/**
	 * The statement violates the constraint.
	 */
	public const STATUS_VIOLATION = 'violation';
	/**
	 * The subject entity of the statement is a known exception to the constraint.
	 */
	public const STATUS_EXCEPTION = 'exception';
	/**
	 * The constraint is not implemented.
	 */
	public const STATUS_TODO = 'todo';
	/**
	 * The constraint parameters are broken.
	 */
	public const STATUS_BAD_PARAMETERS = 'bad-parameters';
	/**
	 * The constraint has not been checked because the statement is deprecated.
	 */
	public const STATUS_DEPRECATED = 'deprecated';
	/**
	 * The statement violates the constraint, but the constraint is not mandatory.
	 *
	 * DelegatingConstraintChecker downgrades violations to warnings automatically based on the constraint parameters;
	 * constraint checkers should not assign this status directly.
	 */
	public const STATUS_WARNING = 'warning';
	/**
	 * The statement violates the constraint, but the constraint is just a suggestion.
	 *
	 * DelegatingConstraintChecker downgrades violations to suggestions automatically based on the constraint parameters;
	 * constraint checkers should not assign this status directly.
	 */
	public const STATUS_SUGGESTION = 'suggestion';
	/**
	 * The constraint is not checked on this kind of snak
	 * (main snak, qualifier or reference),
	 * so the constraint check is skipped.
	 */
	public const STATUS_NOT_IN_SCOPE = 'not-in-scope';
	/*
	 * When adding another status, don’t forget to also do the following:
	 * * define messages for it in i18n/:
	 *   * wbqc-constraintreport-status-
	 *   * apihelp-wbcheckconstraints-paramvalue-status-
	 * * declare a color for it in modules/SpecialConstraintReportPage.less
	 * * update $order in DelegatingConstraintChecker::sortResult
	 * * update PARAM_STATUS type in CheckConstraints::getAllowedParams
	 */

	/** @var Constraint */
	private Constraint $constraint;

	private ContextCursor $contextCursor;

	/**
	 * @var string One of the self::STATUS_… constants
	 */
	private string $status;

	private ?ViolationMessage $message;

	private Metadata $metadata;

	private ?string $snakType;

	private ?DataValue $dataValue;

	private MultilingualTextValue $constraintClarification;

	/**
	 * @param Context|ContextCursor $contextCursor
	 * @param Constraint $constraint
	 * @param string $status One of the self::STATUS_… constants
	 * @param ViolationMessage|null $message
	 */
	public function __construct(
		$contextCursor,
		Constraint $constraint,
		string $status = self::STATUS_TODO,
		ViolationMessage $message = null
	) {
		if ( $contextCursor instanceof Context ) {
			$context = $contextCursor;
			$this->contextCursor = $context->getCursor();
			$this->snakType = $context->getSnak()->getType();
			$mainSnak = $context->getSnak();
			if ( $mainSnak instanceof PropertyValueSnak ) {
				$this->dataValue = $mainSnak->getDataValue();
			} else {
				$this->dataValue = null;
			}
		} else {
			$this->contextCursor = $contextCursor;
			$this->snakType = null;
			$this->dataValue = null;
		}
		$this->constraint = $constraint;
		$this->status = $status;
		$this->message = $message;
		$this->metadata = Metadata::blank();
		$this->constraintClarification = new MultilingualTextValue( [] );
	}

	public function getContextCursor(): ContextCursor {
		return $this->contextCursor;
	}

	/**
	 * @return string|null only available if the CheckResult was created from a full Context
	 */
	public function getSnakType(): ?string {
		return $this->snakType;
	}

	/**
	 * @return DataValue|null only available if the CheckResult was created from a full Context
	 */
	public function getDataValue(): ?DataValue {
		return $this->dataValue;
	}

	public function getConstraint(): Constraint {
		return $this->constraint;
	}

	public function getConstraintId(): string {
		return $this->constraint->getConstraintId();
	}

	/**
	 * @return string One of the self::STATUS_… constants
	 */
	public function getStatus(): string {
		return $this->status;
	}

	public function setStatus( string $status ): void {
		$this->status = $status;
	}

	public function getMessage(): ?ViolationMessage {
		return $this->message;
	}

	public function setMessage( ?ViolationMessage $message ) {
		$this->message = $message;
	}

	public function withMetadata( Metadata $metadata ): self {
		$this->metadata = $metadata;
		return $this;
	}

	public function getMetadata(): Metadata {
		return $this->metadata;
	}

	public function getConstraintClarification(): MultilingualTextValue {
		return $this->constraintClarification;
	}

	public function setConstraintClarification( MultilingualTextValue $constraintClarification ) {
		$this->constraintClarification = $constraintClarification;
	}

}
