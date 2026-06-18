import NodeCache from "node-cache";

/**
 * In-memory cache shared across API routes. Entries expire after 15 minutes.
 */
const cache = new NodeCache({ stdTTL: 900, checkperiod: 120 });

export default cache;
