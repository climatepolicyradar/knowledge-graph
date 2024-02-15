<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use Wikibase\DataModel\Statement\Statement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ConflictsWithChecker implements ConstraintChecker {

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	/**
	 * @var ConnectionCheckerHelper
	 */
	private $connectionCheckerHelper;

	public function __construct(
		ConstraintParameterParser $constraintParameterParser,
		ConnectionCheckerHelper $connectionCheckerHelper
	) {
		$this->constraintParameterParser = $constraintParameterParser;
		$this->connectionCheckerHelper = $connectionCheckerHelper;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getSupportedContextTypes() {
		return [
			Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
			// TODO T175562
			Context::TYPE_QUALIFIER => CheckResult::STATUS_TODO,
			Context::TYPE_REFERENCE => CheckResult::STATUS_TODO,
		];
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getDefaultContextTypes() {
		return [
			Context::TYPE_STATEMENT,
			// TODO T175562
			// Context::TYPE_QUALIFIER,
			// Context::TYPE_REFERENCE,
		];
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * Checks 'Conflicts with' constraint.
	 *
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @throws ConstraintParameterException
	 * @return CheckResult
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		if ( $context->getSnakRank() === Statement::RANK_DEPRECATED ) {
			return new CheckResult( $context, $constraint, CheckResult::STATUS_DEPRECATED );
		}

		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();

		$propertyId = $this->constraintParameterParser->parsePropertyParameter(
			$constraintParameters,
			$constraintTypeItemId
		);

		$items = $this->constraintParameterParser->parseItemsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			false
		);

		$statementList = $context->getEntity()
			->getStatements()
			->getByRank( [ Statement::RANK_PREFERRED, Statement::RANK_NORMAL ] );

		/*
		 * 'Conflicts with' can be defined with
		 *   a) a property only
		 *   b) a property and a number of items (each combination of property and item forming an individual claim)
		 */
		if ( $items === [] ) {
			$offendingStatement = $this->connectionCheckerHelper->findStatementWithProperty(
				$statementList,
				$propertyId
			);
			if ( $offendingStatement !== null ) {
				$message = ( new ViolationMessage( 'wbqc-violation-message-conflicts-with-property' ) )
					->withEntityId( $context->getSnak()->getPropertyId(), Role::CONSTRAINT_PROPERTY )
					->withEntityId( $propertyId, Role::PREDICATE );
				$status = CheckResult::STATUS_VIOLATION;
			} else {
				$message = null;
				$status = CheckResult::STATUS_COMPLIANCE;
			}
		} else {
			$offendingStatement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
				$statementList,
				$propertyId,
				$items
			);
			if ( $offendingStatement !== null ) {
				$offendingValue = ItemIdSnakValue::fromSnak( $offendingStatement->getMainSnak() );
				$message = ( new ViolationMessage( 'wbqc-violation-message-conflicts-with-claim' ) )
					->withEntityId( $context->getSnak()->getPropertyId(), Role::CONSTRAINT_PROPERTY )
					->withEntityId( $propertyId, Role::PREDICATE )
					->withItemIdSnakValue( $offendingValue, Role::OBJECT );
				$status = CheckResult::STATUS_VIOLATION;
			} else {
				$message = null;
				$status = CheckResult::STATUS_COMPLIANCE;
			}
		}

		return new CheckResult( $context, $constraint, $status, $message );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parsePropertyParameter(
				$constraintParameters,
				$constraintTypeItemId
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		try {
			$this->constraintParameterParser->parseItemsParameter(
				$constraintParameters,
				$constraintTypeItemId,
				false
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

}
