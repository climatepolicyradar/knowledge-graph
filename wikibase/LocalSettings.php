<?php

# Stop anonymous users from creating accounts or editing pages - only admin users should 
# be able to create new accounts, and signed-in users should be able to edit pages
$wgGroupPermissions['*']['edit'] = false;
$wgGroupPermissions['*']['createaccount'] = false;


# The following extensions are added because we have specific needs:
# we use WikibaseQualityConstraints to define inverse and symmetric constraints on our 
# properties, for hierarchical and related properties respectively
wfLoadExtension( 'WikibaseQualityConstraints' );

# These ones are standard in the list of extensions that are included in wikibase.cloud
# see https://www.wbstack.com/users/wiki.html#wiki
# The rest of the items in that list are unnecessary for our purposes
wfLoadExtension( 'Echo');
wfLoadExtension( 'MobileFrontend');
wfLoadExtension( 'RevisionSlider');
wfLoadExtension( 'TwoColConflict');
wfLoadExtension( 'WikiEditor');
