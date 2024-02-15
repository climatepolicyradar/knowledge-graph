<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

/**
 * @license GPL-2.0-or-later
 */
class TooManySparqlRequestsException extends SparqlHelperException {

	public function __construct() {
		parent::__construct( 'The SPARQL query endpoint reports too many requests.' );
	}

}
