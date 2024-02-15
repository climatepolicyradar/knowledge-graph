<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck;

use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\SparqlHelperException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;

/**
 * Checks a constraint on some constraint checking context.
 * Most implementations only support one constraint type.
 *
 * @license GPL-2.0-or-later
 */
interface ConstraintChecker {

	/**
	 * Convenience constant, returned by most {@link getSupportedEntityTypes} implementations.
	 */
	public const ALL_ENTITY_TYPES_SUPPORTED = [
		'item' => CheckResult::STATUS_COMPLIANCE,
		'property' => CheckResult::STATUS_COMPLIANCE,
		'lexeme' => CheckResult::STATUS_COMPLIANCE,
		'form' => CheckResult::STATUS_COMPLIANCE,
		'sense' => CheckResult::STATUS_COMPLIANCE,
		'mediainfo' => CheckResult::STATUS_COMPLIANCE,
	];

	/**
	 * Convenience constant, returned by many {@link getSupportedContextTypes} implementations.
	 */
	public const ALL_CONTEXT_TYPES_SUPPORTED = [
		Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
		Context::TYPE_QUALIFIER => CheckResult::STATUS_COMPLIANCE,
		Context::TYPE_REFERENCE => CheckResult::STATUS_COMPLIANCE,
	];

	/**
	 * Determines which context types this constraint type supports.
	 * checkConstraint() should only be called for contexts with one of the supported types.
	 *
	 * Returns an array from context types
	 * (i. e., Context::TYPE_* constants)
	 * to result status (i. e., CheckResult::STATUS_* constants).
	 * STATUS_COMPLIANCE means that the constraint type supports this context type
	 * (checkConstraint() might of course return a different status, e. g. VIOLATION);
	 * STATUS_TODO means that the constraint type might support this context type in the future,
	 * but it is not currently supported;
	 * and STATUS_NOT_IN_SCOPE means that the constraint type does not support this context type.
	 *
	 * For example, the array
	 *
	 *     [
	 *         Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
	 *         Context::TYPE_QUALIFIER => CheckResult::STATUS_TODO,
	 *         Context::TYPE_REFERENCE => CheckResult::STATUS_NOT_IN_SCOPE,
	 *     ]
	 *
	 * indicates that a constraint type makes sense on statements and qualifiers
	 * (but not references), but has only been implemented on statements so far.
	 *
	 * Many implementations can just return {@link ALL_CONTEXT_TYPES_SUPPORTED}.
	 *
	 * @return string[]
	 */
	public function getSupportedContextTypes();

	/**
	 * Determines the context types where this constraint type is checked
	 * if the constraint scope has not been explicitly specified as a constraint parameter.
	 * Returns an array of context types (i. e., Context::TYPE_* constants).
	 *
	 * For example, the array [ Context::TYPE_STATEMENT ] indicates that,
	 * by default, a constraint should only be checked on the main snak of a statement.
	 * Depending on the {@link getSupportedContextTypes supported context types},
	 * it might also be checked on other context types
	 * if the constraint explicitly specifies a different scope
	 * (which might not even include the “statement” scope).
	 *
	 * Many implementations can just return {@link Context::ALL_CONTEXT_TYPES}.
	 *
	 * @return string[]
	 */
	public function getDefaultContextTypes();

	/**
	 * Determines which entity types this constraint type supports.
	 * checkConstraint() should only be called for contexts with one of the supported entity types.
	 *
	 * Returns an array from entity types to result status (CheckResult::STATUS_* constants).
	 * The meaning of STATUS_COMPLIANCE, STATUS_TODO and STATUS_NOT_IN_SCOPE
	 * is the same as for {@link getSupportedContextTypes}.
	 *
	 * Most implementations can just return {@link ALL_ENTITY_TYPES_SUPPORTED}.
	 *
	 * @return string[]
	 */
	public function getSupportedEntityTypes();

	/**
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @return CheckResult
	 *
	 * @throws ConstraintParameterException if the constraint parameters are invalid
	 * @throws SparqlHelperException if the checker uses SPARQL and the query times out or some other error occurs
	 */
	public function checkConstraint( Context $context, Constraint $constraint );

	/**
	 * Check if the constraint parameters of $constraint are valid.
	 * Returns a list of ConstraintParameterExceptions, one for each problematic parameter;
	 * if the list is empty, all constraint parameters are okay.
	 *
	 * @param Constraint $constraint
	 *
	 * @return ConstraintParameterException[]
	 */
	public function checkConstraintParameters( Constraint $constraint );

}
