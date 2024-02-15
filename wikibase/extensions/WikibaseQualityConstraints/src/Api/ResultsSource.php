<?php

namespace WikibaseQuality\ConstraintReport\Api;

use Wikibase\DataModel\Entity\EntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;

/**
 * A source of constraint check results for a given constraint checking request.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
interface ResultsSource {

	/**
	 * @param EntityId[] $entityIds entity IDs to check
	 * @param string[] $claimIds statement IDs to check
	 * @param ?string[] $constraintIds if not null, limit checks to these constraint IDs
	 * @param string[] $statuses return only results with these statuses
	 * @return CachedCheckResults
	 */
	public function getResults(
		array $entityIds,
		array $claimIds,
		?array $constraintIds,
		array $statuses
	);

}
