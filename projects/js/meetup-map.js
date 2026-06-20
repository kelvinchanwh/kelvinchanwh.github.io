(function () {
	const maxLocations = 6;
	let map;
	let geocoder;
	let placeSearcher = null;
	let routeClass = null;
	let routePolylines = [];
	let markers = [];
	let centerMarker = null;
	let areaCircle = null;
	let autocompletes = [];
	let previewDebounceTimer = null;
	let hasMapsAuthFailure = false;
	const originColors = ['#d93025', '#1a73e8', '#188038', '#f9ab00', '#9334e6', '#00897b'];
	const WALKING_METERS_PER_MINUTE = 80;
	const MAX_WALK_MINUTES_WITHIN_AREA = 5;
	const MAX_WALKABLE_AREA_RADIUS_METERS = WALKING_METERS_PER_MINUTE * MAX_WALK_MINUTES_WITHIN_AREA;

	function $(selector) {
		return document.querySelector(selector);
	}

	function getOriginColor(index) {
		return originColors[index % originColors.length];
	}

	function createLocationRow(index) {
		const wrapper = document.createElement('div');
		wrapper.className = 'location-row';

		const label = document.createElement('label');
		label.textContent = `Location ${index}`;

		const input = document.createElement('input');
		input.name = 'location';
		input.type = 'text';
		input.placeholder = 'Enter address or place';
		input.setAttribute('aria-label', `Location ${index}`);

		wrapper.appendChild(label);
		wrapper.appendChild(input);
		return wrapper;
	}

	function createAutocomplete() {
		if (hasMapsAuthFailure) {
			throw new Error('Google Maps authentication failed. Autocomplete is disabled.');
		}
		if (!google?.maps?.places?.PlaceAutocompleteElement) {
			throw new Error('PlaceAutocompleteElement is not available. Please load the current Google Maps Places API.');
		}

		const options = {};

		return new google.maps.places.PlaceAutocompleteElement(options);
	}

	function addLocationInput() {
		const locationList = $('#location-list');
		const count = locationList.querySelectorAll('.location-row').length;
		if (count >= maxLocations) {
			alert('You can add up to ' + maxLocations + ' locations.');
			return;
		}

		const newRow = createLocationRow(count + 1);
		locationList.appendChild(newRow);
		const newInput = newRow.querySelector('input');
		if (!hasMapsAuthFailure) {
			try {
				const autocomplete = createAutocomplete();
				newInput.parentNode.replaceChild(autocomplete, newInput);
				autocompletes.push(autocomplete);
				bindAutocompleteEvents(autocomplete);
				try {
					autocomplete.focus();
				} catch (e) {
					// ignore
				}
			} catch (error) {
				console.warn('Autocomplete disabled for this row:', error?.message || error);
			}
		}
		scheduleMapPreviewUpdate();
	}

	function getOrigins() {
		return autocompletes.map(ac => ac.value.trim()).filter(Boolean);
	}

	function bindAutocompleteEvents(autocomplete) {
		autocomplete.addEventListener('gmp-select', scheduleMapPreviewUpdate);
		autocomplete.addEventListener('change', scheduleMapPreviewUpdate);
	}

	function scheduleMapPreviewUpdate() {
		clearTimeout(previewDebounceTimer);
		previewDebounceTimer = setTimeout(updateMapPreview, 250);
	}

	async function updateMapPreview() {
		if (!map || !geocoder) return;
		const origins = getOrigins();
		clearMarkers();

		if (!origins.length) {
			map.setCenter({ lat: 1.3521, lng: 103.8198 });
			map.setZoom(2);
			return;
		}

		const geoPoints = (await Promise.all(origins.map((origin) => geocodeAddress(origin).catch(() => null))))
			.filter(Boolean);

		if (!geoPoints.length) {
			map.setCenter({ lat: 1.3521, lng: 103.8198 });
			map.setZoom(2);
			return;
		}

		if (geoPoints.length === 1) {
			map.setCenter(geoPoints[0]);
			map.setZoom(12);
			createMarker(geoPoints[0], 'Origin 1', '1', 'location', getOriginColor(0));
			return;
		}

		const bounds = new google.maps.LatLngBounds();
		geoPoints.forEach((point, index) => {
			bounds.extend(point);
			createMarker(point, `Origin ${index + 1}`, `${index + 1}`, 'location', getOriginColor(index));
		});
		map.fitBounds(bounds);
	}

	function clearMarkers() {
		markers.forEach((marker) => marker.map = null);
		markers = [];
		if (centerMarker) {
			centerMarker.map = null;
			centerMarker = null;
		}
		if (areaCircle) {
			areaCircle.setMap(null);
			areaCircle = null;
		}
		if (routePolylines.length) {
			routePolylines.forEach((polyline) => polyline.setMap(null));
			routePolylines = [];
		}
	}

	function buildMarkerContent(label, variant = 'location') {
		const markerLabel = document.createElement('span');
		markerLabel.className = `meetup-marker meetup-marker-${variant}`;
		markerLabel.textContent = label || '';
		return markerLabel;
	}

	function createMarker(position, title, label, variant = 'location', color = null) {
		if (!map) return null;
		const content = buildMarkerContent(label, variant);
		if (variant === 'location' && color) {
			content.style.setProperty('--origin-color', color);
		}
		const marker = new google.maps.marker.AdvancedMarkerElement({
			position,
			map,
			title,
			content,
		});
		markers.push(marker);
		return marker;
	}

	function geocodeAddress(address) {
		return new Promise((resolve, reject) => {
			geocoder.geocode({ address }, (results, status) => {
				if (status === 'OK' && results[0]) {
					resolve(results[0].geometry.location);
				} else {
					reject(new Error(`Unable to geocode "${address}" (${status})`));
				}
			});
		});
	}

	function computeCentroid(points) {
		const sum = points.reduce(
			(acc, point) => {
				acc.lat += point.lat();
				acc.lng += point.lng();
				return acc;
			},
			{ lat: 0, lng: 0 }
		);

		return {
			lat: sum.lat / points.length,
			lng: sum.lng / points.length,
		};
	}

	function toLatLngLiteral(point) {
		if (!point) return { lat: 0, lng: 0 };
		if (typeof point.lat === 'function' && typeof point.lng === 'function') {
			return { lat: point.lat(), lng: point.lng() };
		}
		return { lat: Number(point.lat), lng: Number(point.lng) };
	}

	function toRoutePoint(latLngLiteral) {
		return {
			lat: () => latLngLiteral.lat,
			lng: () => latLngLiteral.lng,
		};
	}

	function haversineDistanceMeters(a, b) {
		const aLiteral = toLatLngLiteral(a);
		const bLiteral = toLatLngLiteral(b);
		const toRad = (value) => value * Math.PI / 180;
		const earthRadiusMeters = 6371000;
		const dLat = toRad(bLiteral.lat - aLiteral.lat);
		const dLng = toRad(bLiteral.lng - aLiteral.lng);
		const lat1 = toRad(aLiteral.lat);
		const lat2 = toRad(bLiteral.lat);
		const sinDlat = Math.sin(dLat / 2);
		const sinDlng = Math.sin(dLng / 2);
		const h = sinDlat * sinDlat + Math.cos(lat1) * Math.cos(lat2) * sinDlng * sinDlng;
		return 2 * earthRadiusMeters * Math.asin(Math.sqrt(h));
	}

	function offsetLatLng(center, northMeters, eastMeters) {
		const base = toLatLngLiteral(center);
		const latDelta = northMeters / 111320;
		const cosLat = Math.cos((base.lat * Math.PI) / 180);
		const safeCosLat = Math.abs(cosLat) < 1e-6 ? 1e-6 : cosLat;
		const lngDelta = eastMeters / (111320 * safeCosLat);
		return {
			lat: base.lat + latDelta,
			lng: base.lng + lngDelta,
		};
	}

	function computeSeedCenter(origins) {
		if (origins.length === 2) {
			return {
				lat: (origins[0].lat() + origins[1].lat()) / 2,
				lng: (origins[0].lng() + origins[1].lng()) / 2,
			};
		}
		return computeCentroid(origins);
	}

	function buildAreaCandidates(origins, seedCenter) {
		const spread = origins.reduce((max, origin) => {
			const distance = haversineDistanceMeters(origin, seedCenter);
			return Math.max(max, distance);
		}, 0);
		const searchRadius = Math.min(Math.max(spread * 0.8, 800), 5000);
		const rings = [0, searchRadius * 0.35, searchRadius * 0.7, searchRadius];
		const directions = [
			{ n: 1, e: 0 },
			{ n: 0.707, e: 0.707 },
			{ n: 0, e: 1 },
			{ n: -0.707, e: 0.707 },
			{ n: -1, e: 0 },
			{ n: -0.707, e: -0.707 },
			{ n: 0, e: -1 },
			{ n: 0.707, e: -0.707 },
		];

		const candidates = [toLatLngLiteral(seedCenter)];
		rings.forEach((ringRadius, index) => {
			if (index === 0) return;
			directions.forEach((direction) => {
				candidates.push(offsetLatLng(seedCenter, direction.n * ringRadius, direction.e * ringRadius));
			});
		});

		return candidates;
	}

	function estimateMetricsByMode(origin, destination, travelMode) {
		const distanceMeters = haversineDistanceMeters(origin, destination);
		const speedByModeKmh = {
			driving: 35,
			walking: 4.8,
			transit: 24,
			bicycling: 15,
		};
		const modeKey = String(travelMode || 'driving').toLowerCase();
		const speedKmh = speedByModeKmh[modeKey] || speedByModeKmh.driving;
		const durationSeconds = (distanceMeters / 1000) / speedKmh * 3600;
		const distanceKm = distanceMeters / 1000;
		const durationMinutes = durationSeconds / 60;
		const costEstimate = distanceKm * 0.2 + durationMinutes * 0.05;
		return {
			distanceMeters,
			durationSeconds,
			costEstimate,
			costDisplay: `~$${costEstimate.toFixed(2)}`,
			distanceKm,
			durationMinutes,
		};
	}

	function getPreferredAreaTypesByRadius(radiusMeters) {
		if (radiusMeters <= 400) {
			return [
				'point_of_interest',
				'transit_station',
				'establishment',
				'neighborhood',
				'sublocality_level_1',
				'sublocality',
				'route',
				'postal_town',
				'locality',
			];
		}
		if (radiusMeters <= 900) {
			return [
				'neighborhood',
				'sublocality_level_1',
				'sublocality',
				'route',
				'postal_town',
				'locality',
			];
		}
		return ['sublocality_level_1', 'sublocality', 'locality', 'postal_town', 'administrative_area_level_2'];
	}

	function extractAreaNameFromGeocodeResults(results, radiusMeters) {
		if (!results?.length) return null;
		const preferredTypes = getPreferredAreaTypesByRadius(radiusMeters);
		for (const type of preferredTypes) {
			for (const result of results) {
				const components = result.address_components || [];
				const match = components.find((component) => component.types?.includes(type));
				if (match?.long_name) {
					return match.long_name;
				}
			}
		}

		if (results[0]?.formatted_address) {
			return results[0].formatted_address.split(',').slice(0, 2).join(',').trim();
		}
		return null;
	}

	function reverseGeocodeAreaName(center, radiusMeters = MAX_WALKABLE_AREA_RADIUS_METERS) {
		if (!geocoder) return Promise.resolve(null);
		return new Promise((resolve) => {
			geocoder.geocode({ location: toLatLngLiteral(center) }, (results, status) => {
				if (status === 'OK' && results?.[0]) {
					resolve(extractAreaNameFromGeocodeResults(results, radiusMeters));
					return;
				}
				resolve(null);
			});
		});
	}

	function normalizePlaceResult(place) {
		const location = place.geometry?.location ?? place.location;
		return {
			...place,
			geometry: { location },
			name: place.displayName || place.name || '',
			vicinity: place.formattedAddress || place.vicinity || place.address || '',
			formatted_address: place.formattedAddress || place.formatted_address || '',
		};
	}

	function loadPlaces(center, category, maxResults = 20) {
		return new Promise(async (resolve, reject) => {
			if (!placeSearcher?.searchByText) {
				reject(new Error('Place search is not available.'));
				return;
			}

			try {
				const request = {
					textQuery: category,
					locationBias: toLatLngLiteral(center),
					includedType: category,
					maxResultCount: maxResults,
					fields: ['displayName', 'formattedAddress', 'location'],
				};
				const result = await placeSearcher.searchByText(request);
				const places = result?.places?.map(normalizePlaceResult) ?? [];
				if (places.length) {
					resolve(places);
				} else {
					reject(new Error('No places found nearby.'));
				}
			} catch (error) {
				reject(new Error(`Place search failed: ${error?.message || error}`));
			}
		});
	}

	function getRoute(origin, destination, travelMode) {
		if (!map) return Promise.resolve({ routes: [] });
		return new Promise((resolve, reject) => {
			if (!routeClass?.computeRoutes) {
				reject(new Error('Routes API is not available.'));
				return;
			}

			const request = {
				origin,
				destination,
				travelMode: google.maps.TravelMode?.[travelMode.toUpperCase()] || travelMode.toUpperCase(),
				fields: ['legs', 'localizedValues', 'travelAdvisory'],
			};

			routeClass.computeRoutes(request)
				.then((result) => {
					const routes = result?.routes ?? [];
					if (routes.length) {
						const route = routes[0];
						if (!route.legs || route.legs.length === 0) {
							console.warn('Routes API returned route without legs data');

						}
						resolve({ routes });
						return;
					}
					reject(new Error(`Routing failed: no routes returned from (${origin.lat()}, ${origin.lng()}) to (${destination.lat()}, ${destination.lng()})`));
				})
				.catch((error) => reject(error));
		});
	}

	function routeScore(metrics, strategy) {
		switch (strategy) {
			case 'travel_time_first':
				return computeTravelTimeFirstScore(metrics);
			case 'distance':
				return metrics.distanceMeters;
			case 'time':
				return metrics.durationSeconds;
			case 'cost':
				return metrics.costEstimate;
			case 'recommended':
			default:
				return metrics.distanceMeters + metrics.durationSeconds + metrics.costEstimate * 100;
		}
	}

	function computeTravelTimeFirstScore(metrics) {
		const total = metrics.durationSeconds ?? 0;
		const imbalance = metrics.timeImbalanceSeconds ?? 0;
		const maxDuration = metrics.maxDurationSeconds ?? total;
		return maxDuration * 3 + total + imbalance * 1.5;
	}

	function parseDurationSeconds(value) {
		if (typeof value === 'number' && Number.isFinite(value)) {
			return value;
		}
		if (typeof value === 'string') {
			const match = value.match(/^([0-9]+(?:\.[0-9]+)?)s$/);
			if (match) {
				return Number(match[1]);
			}
		}
		if (value && typeof value === 'object') {
			const seconds = Number(value.seconds ?? 0);
			const nanos = Number(value.nanos ?? 0);
			if (Number.isFinite(seconds) || Number.isFinite(nanos)) {
				return seconds + nanos / 1e9;
			}
		}
		return 0;
	}

	function parseDistanceMeters(value) {
		if (typeof value === 'number' && Number.isFinite(value)) {
			return value;
		}
		if (value && typeof value === 'object') {
			const numeric = Number(value.value ?? 0);
			if (Number.isFinite(numeric)) {
				return numeric;
			}
		}
		return 0;
	}

	function drawRoute(route, append = false, color = null) {
		if (!map) return;
		if (!append && routePolylines.length) {
			routePolylines.forEach((polyline) => polyline.setMap(null));
			routePolylines = [];
		}
		if (!route || typeof route.createPolylines !== 'function') return;
		try {
			const polylines = route.createPolylines();
			polylines.forEach((polyline) => {
				if (color && typeof polyline.setOptions === 'function') {
					polyline.setOptions({
						strokeColor: color,
						strokeOpacity: 0.9,
						strokeWeight: 5,
					});
				}
				polyline.setMap(map);
			});
			routePolylines.push(...polylines);
		} catch (error) {
			console.warn('Failed to draw route polyline:', error);
		}
	}

	function selectShortestRoute(routes, strategy) {
		if (!routes || !routes.length) return null;
		return routes.reduce((best, current) => {
			const bestScore = routeScore(routeMetrics({ routes: [best] }), strategy);
			const currentScore = routeScore(routeMetrics({ routes: [current] }), strategy);
			return currentScore < bestScore ? current : best;
		});
	}

	async function drawShortestRoutes(origins, destination, travelMode, strategy) {
		if (routePolylines.length) {
			routePolylines.forEach((polyline) => polyline.setMap(null));
			routePolylines = [];
		}

		const routeCandidates = await Promise.all(origins.map((origin) => getRoute(origin, destination, travelMode).catch(() => null)));
		routeCandidates.forEach((candidate, index) => {
			if (!candidate?.routes?.length) return;
			const shortestRoute = selectShortestRoute(candidate.routes, strategy);
			if (shortestRoute) {
				drawRoute(shortestRoute, true, getOriginColor(index));
			}
		});
	}

	function formatMoney(money) {
		if (!money) return null;
		const units = Number(money.units ?? 0);
		const nanos = Number(money.nanos ?? 0);
		const amount = units + nanos / 1e9;
		const currency = money.currencyCode || '';
		try {
			return new Intl.NumberFormat(undefined, { style: 'currency', currency }).format(amount);
		} catch {
			return `${currency} ${amount.toFixed(2)}`;
		}
	}

	function routeMetrics(response) {
		if (!response || !response.routes || response.routes.length === 0) {
			return { distanceMeters: 0, durationSeconds: 0, costDisplay: null, distanceKm: 0, durationMinutes: 0 };
		}
		const route = response.routes[0];

		// JS SDK uses durationMillis (number), not duration string "Xs" like the REST API
		const leg = route.legs?.[0];
		const durationMillis = leg?.durationMillis ?? route.durationMillis ?? 0;
		const distanceMeters = parseDistanceMeters(leg?.distanceMeters ?? route.distanceMeters);
		const durationSeconds = Number(durationMillis) / 1000;
		const distanceKm = distanceMeters / 1000;
		const durationMinutes = durationSeconds / 60;

		// Cost: use API-provided fare/toll if available; JS SDK property is estimatedPrices (plural)
		const advisory = route.travelAdvisory;
		const transitFare = advisory?.transitFare;
		const tollPrice = advisory?.tollInfo?.estimatedPrices?.[0];
		// localizedValues.transitFare is a pre-formatted string (e.g. "$2.25")
		const transitFareText = route.localizedValues?.transitFare;
		let costDisplay = null;
		let costEstimate = 0;
		if (transitFareText) {
			costDisplay = transitFareText;
			costEstimate = Number(transitFare?.units ?? 0) + Number(transitFare?.nanos ?? 0) / 1e9;
		} else if (tollPrice) {
			costDisplay = formatMoney(tollPrice);
			costEstimate = Number(tollPrice.units ?? 0) + Number(tollPrice.nanos ?? 0) / 1e9;
		} else {
			costEstimate = distanceKm * 0.2 + durationMinutes * 0.05;
			costDisplay = `~$${costEstimate.toFixed(2)}`;
		}

		return { distanceMeters, durationSeconds, costEstimate, costDisplay, distanceKm, durationMinutes };
	}

	function selectBestPlace(candidates, strategy) {
		return candidates.reduce((best, current) => {
			const score = current.scores[strategy];
			if (best === null || score < best.scores[strategy]) {
				return current;
			}
			return best;
		}, null);
	}

	function selectTopPlaces(candidates, strategy, count = 5) {
		const effectiveStrategy = strategy || 'travel_time_first';
		return [...candidates]
			.sort((a, b) => {
				const scoreDelta = (a.scores[effectiveStrategy] ?? Infinity) - (b.scores[effectiveStrategy] ?? Infinity);
				if (scoreDelta !== 0) return scoreDelta;
				return (a.place.name || '').localeCompare(b.place.name || '');
			})
			.slice(0, count);
	}

	async function evaluateAreaCandidates(candidates, origins, travelMode) {
		const scoredCandidates = [];
		for (const candidateCenter of candidates) {
			const destination = toRoutePoint(candidateCenter);
			const routePromises = origins.map((origin, index) => getRoute(origin, destination, travelMode)
				.then((route) => ({ index, route }))
				.catch(() => null));
			const routesWithIndex = await Promise.all(routePromises);
			const routeByIndex = new Map(routesWithIndex.filter(Boolean).map((entry) => [entry.index, entry.route]));
			const perOrigin = origins.map((origin, index) => {
				const route = routeByIndex.get(index);
				if (route) {
					return { originIndex: index, ...routeMetrics(route) };
				}
				return { originIndex: index, ...estimateMetricsByMode(origin, candidateCenter, travelMode) };
			});

			const totals = perOrigin.reduce(
				(acc, metrics) => {
					acc.distanceMeters += metrics.distanceMeters;
					acc.durationSeconds += metrics.durationSeconds;
					acc.costEstimate += metrics.costEstimate;
					return acc;
				},
				{ distanceMeters: 0, durationSeconds: 0, costEstimate: 0 }
			);

			const durations = perOrigin.map((originMetric) => originMetric.durationSeconds);
			const maxDuration = Math.max(...durations);
			const minDuration = Math.min(...durations);
			const timeImbalanceSeconds = maxDuration - minDuration;
			const travelTimeScore = computeTravelTimeFirstScore({
				durationSeconds: totals.durationSeconds,
				timeImbalanceSeconds,
				maxDurationSeconds: maxDuration,
			});
			const maxDistanceFromCenter = origins.reduce((maxDistance, origin) => {
				const distance = haversineDistanceMeters(origin, candidateCenter);
				return Math.max(maxDistance, distance);
			}, 0);

			scoredCandidates.push({
				center: toLatLngLiteral(candidateCenter),
				perOrigin,
				totals,
				timeImbalanceSeconds,
				maxDurationSeconds: maxDuration,
				score: travelTimeScore,
				radiusMeters: Math.min(Math.max(maxDistanceFromCenter * 0.12, 180), MAX_WALKABLE_AREA_RADIUS_METERS),
			});
		}

		return scoredCandidates;
	}

	function selectBestArea(scoredAreas) {
		if (!scoredAreas.length) return null;
		return scoredAreas.reduce((best, current) => current.score < best.score ? current : best);
	}

	function isPlaceInsideArea(place, area) {
		const placeLocation = place?.geometry?.location;
		if (!placeLocation || !area) return false;
		return haversineDistanceMeters(placeLocation, area.center) <= area.radiusMeters;
	}

	async function evaluatePlaces(places, origins, travelMode) {
		const candidates = [];
		for (const place of places) {
			const routePromises = origins.map((origin, index) => getRoute(origin, place.geometry.location, travelMode)
				.then((route) => ({ index, route }))
				.catch((error) => {
					console.warn(`Route failed from ${origin.lat()}, ${origin.lng()} to ${place.geometry.location.lat()}, ${place.geometry.location.lng()}:`, error.message);
					return null;
				}));
			const routesWithIndex = await Promise.all(routePromises);
			const validRoutes = routesWithIndex.filter((entry) => entry !== null);
			if (validRoutes.length === 0) {
				continue;
			}

			const perOrigin = validRoutes
				.sort((a, b) => a.index - b.index)
				.map((entry) => {
					const metrics = routeMetrics(entry.route);
					return { originIndex: entry.index, ...metrics };
				});

			const totals = perOrigin.reduce(
				(acc, m) => {
					acc.distanceMeters += m.distanceMeters;
					acc.durationSeconds += m.durationSeconds;
					acc.costEstimate += m.costEstimate;
					return acc;
				},
				{ distanceMeters: 0, durationSeconds: 0, costEstimate: 0 }
			);

			const durations = perOrigin.map((m) => m.durationSeconds);
			const maxDurationSeconds = Math.max(...durations);
			const timeImbalanceSeconds = Math.max(...durations) - Math.min(...durations);

			candidates.push({
				place,
				perOrigin,
				totals,
				maxDurationSeconds,
				scores: {
					travel_time_first: computeTravelTimeFirstScore({
						durationSeconds: totals.durationSeconds,
						timeImbalanceSeconds,
						maxDurationSeconds,
					}),
					distance: totals.distanceMeters,
					time: totals.durationSeconds,
					cost: totals.costEstimate,
					recommended: totals.distanceMeters + totals.durationSeconds + totals.costEstimate * 100,
				},
			});
		}
		return candidates;
	}

	function formatDuration(seconds) {
		const minutes = Math.round(seconds / 60);
		if (minutes < 60) {
			return `${minutes} min`;
		}
		const hours = Math.floor(minutes / 60);
		const remainder = minutes % 60;
		return `${hours} hr ${remainder} min`;
	}

	function formatDistance(meters) {
		const roundedMeters = Math.round(meters);
		if (meters >= 1000) {
			return `${(meters / 1000).toFixed(1)} km`;
		}
		return `${roundedMeters} m`;
	}

	function renderResults(area, rankedPlaces, origins, areaName) {
		const fallbackAreaName = rankedPlaces[0]?.place?.vicinity || rankedPlaces[0]?.place?.formatted_address || 'Selected meetup area';
		const resolvedAreaName = areaName || fallbackAreaName;
		const areaRows = area.perOrigin
			.map((metrics) => {
				const originLabel = origins[metrics.originIndex] || `Origin ${metrics.originIndex + 1}`;
				return `<tr>
				<td>${originLabel}</td>
				<td>${formatDuration(metrics.durationSeconds)}</td>
				<td>${formatDistance(metrics.distanceMeters)}</td>
			</tr>`;
			})
			.join('');

		const placeRows = rankedPlaces
			.map((candidate, index) => `<tr>
				<td>${index + 1}</td>
				<td>${candidate.place.name}</td>
				<td>${formatDuration(candidate.totals.durationSeconds)}</td>
				<td>${formatDistance(candidate.totals.distanceMeters)}</td>
				<td>${candidate.place.vicinity || candidate.place.formatted_address || ''}</td>
			</tr>`)
			.join('');

		const html = `
			<h3>Best meetup area</h3>
			<p>Area: ${resolvedAreaName} · Radius: ${Math.round(area.radiusMeters)} m (within ~${MAX_WALK_MINUTES_WITHIN_AREA} min walk)</p>
			<p>Combined travel time: ${formatDuration(area.totals.durationSeconds)} · Fairness gap: ${formatDuration(area.timeImbalanceSeconds)}</p>
			<table>
				<thead>
					<tr>
						<th>Origin</th>
						<th>Travel time to area</th>
						<th>Distance to area</th>
					</tr>
				</thead>
				<tbody>
					${areaRows}
				</tbody>
			</table>
			<h3>Top ${rankedPlaces.length} options in this area</h3>
			<table>
				<thead>
					<tr>
						<th>#</th>
						<th>Place</th>
						<th>Total travel time</th>
						<th>Total distance</th>
						<th>Address</th>
					</tr>
				</thead>
				<tbody>
					${placeRows}
				</tbody>
			</table>
		`;

		$('#results-container').innerHTML = html;
	}

	function renderMap(origins, area, rankedPlaces) {
		if (!map) return;
		clearMarkers();
		map.setCenter(area.center);
		map.setZoom(11);

		origins.forEach((origin, index) => {
			const marker = createMarker(origin, `Origin ${index + 1}`, `${index + 1}`, 'location', getOriginColor(index));
			if (!marker) return;
			const infowindow = new google.maps.InfoWindow({ content: `Origin ${index + 1}` });
			marker.addEventListener('gmp-click', () => infowindow.open({ map, anchor: marker }));
		});

		areaCircle = new google.maps.Circle({
			strokeColor: '#1a73e8',
			strokeOpacity: 0.9,
			strokeWeight: 2,
			fillColor: '#1a73e8',
			fillOpacity: 0.14,
			map,
			center: area.center,
			radius: area.radiusMeters,
		});

		rankedPlaces.forEach((candidate, index) => {
			const label = `${index + 1}`;
			const marker = createMarker(candidate.place.geometry.location, candidate.place.name, label, 'best-place');
			if (!marker) return;
			const infoContent = `<strong>${index + 1}. ${candidate.place.name}</strong><br>${candidate.place.vicinity || ''}<br>Total travel time: ${formatDuration(candidate.totals.durationSeconds)}`;
			const infowindow = new google.maps.InfoWindow({ content: infoContent });
			marker.addEventListener('gmp-click', () => infowindow.open({ map, anchor: marker }));
		});

		const bounds = new google.maps.LatLngBounds();
		origins.forEach((origin) => bounds.extend(origin));
		rankedPlaces.forEach((candidate) => bounds.extend(candidate.place.geometry.location));
		bounds.extend(area.center);
		map.fitBounds(bounds);
	}

	async function optimizeMeetup() {
		const origins = getOrigins();
		const category = $('#poi-category').value;
		const travelMode = $('#transport-mode').value;
		const strategy = $('#optimization-strategy').value || 'travel_time_first';

		const results = $('#results-container');
		results.innerHTML = '<p>Computing the best meetup option... please wait.</p>';

		if (origins.length < 2) {
			results.innerHTML = '<p>Please enter at least two locations.</p>';
			return;
		}

		try {
			const geoPoints = await Promise.all(origins.map(geocodeAddress));
			const seedCenter = computeSeedCenter(geoPoints);
			const areaCandidates = buildAreaCandidates(geoPoints, seedCenter);
			const scoredAreas = await evaluateAreaCandidates(areaCandidates, geoPoints, travelMode);
			const bestArea = selectBestArea(scoredAreas);
			if (!bestArea) {
				throw new Error('Unable to determine a meetup area from these origins.');
			}
			const placeCandidates = await loadPlaces(bestArea.center, category, 20);
			const placeInArea = placeCandidates.filter((place) => isPlaceInsideArea(place, bestArea));
			if (!placeInArea.length) {
				throw new Error('No places were found inside the recommended area.');
			}
			const evaluatedPlaces = await evaluatePlaces(placeInArea, geoPoints, travelMode);
			const rankedPlaces = selectTopPlaces(evaluatedPlaces, strategy, 5);
			if (!rankedPlaces.length) {
				throw new Error('Unable to rank places in the recommended area.');
			}
			const areaName = await reverseGeocodeAreaName(bestArea.center, bestArea.radiusMeters);
			renderMap(geoPoints, bestArea, rankedPlaces);
			await drawShortestRoutes(geoPoints, rankedPlaces[0].place.geometry.location, travelMode, strategy);
			renderResults(bestArea, rankedPlaces, origins, areaName);
		} catch (error) {
			results.innerHTML = `<p class="error">${error.message}</p>`;
		}
	}

	function resetForm() {
		$('#results-container').innerHTML = '';
		clearMarkers();
	}

	function attachEvents() {
		$('#add-location-button').addEventListener('click', addLocationInput);
		$('#optimize-button').addEventListener('click', optimizeMeetup);
		$('#reset-button').addEventListener('click', resetForm);
	}

	window.initMeetupMap = function () {
		if (hasMapsAuthFailure) {
			const results = $('#results-container');
			if (results) {
				results.innerHTML = '<p class="error">Google Maps authentication failed. Please fix your API key and reload the page.</p>';
			}
			return;
		}
		const mapElement = document.getElementById('meetup-map-canvas');
		map = new google.maps.Map(mapElement, {
			center: { lat: 1.3521, lng: 103.8198 },
			zoom: 2,
			mapId: 'DEMO_MAP_ID',
			disableDefaultUI: true,
		});
		geocoder = new google.maps.Geocoder();
		if (google?.maps?.places?.Place?.searchByText) {
			placeSearcher = google.maps.places.Place;
		}

		if (!placeSearcher) {
			alert('Places API is not available. Please ensure the Places API (New) is enabled and loaded.');
		}

		if (google.maps.routes?.Route) {
			routeClass = google.maps.routes.Route;
		}

		if (!routeClass) {
			alert('Routes API is not available. Please ensure the Routes API is enabled and loaded.');
		}

		// Initialize autocomplete for existing location inputs
		document.querySelectorAll('input[name="location"]').forEach(input => {
			try {
				const autocomplete = createAutocomplete();
				input.parentNode.replaceChild(autocomplete, input);
				autocompletes.push(autocomplete);
				bindAutocompleteEvents(autocomplete);
			} catch (error) {
				console.warn('Failed to initialize autocomplete for input:', error?.message || error);
			}
		});

		scheduleMapPreviewUpdate();

		attachEvents();
	};

	window.gm_authFailure = function () {
		hasMapsAuthFailure = true;
		autocompletes = [];
		const results = $('#results-container');
		const message = 'Google Maps authentication failed. Please verify your API key and restrictions.';
		if (results) {
			results.innerHTML = `<p class="error">${message}</p>`;
		} else {
			console.error(message);
		}
	};

	// Export functions for testing
	window.meetupMap = {
		geocodeAddress,
		computeCentroid,
		computeSeedCenter,
		buildAreaCandidates,
		evaluateAreaCandidates,
		selectBestArea,
		getPreferredAreaTypesByRadius,
		extractAreaNameFromGeocodeResults,
		reverseGeocodeAreaName,
		isPlaceInsideArea,
		loadPlaces,
		getRoute,
		routeMetrics,
		computeTravelTimeFirstScore,
		selectBestPlace,
		selectTopPlaces,
		evaluatePlaces,
		formatDuration,
		formatDistance,
		renderResults,
		renderMap,
		optimizeMeetup,
		resetForm,
		attachEvents,
		addLocationInput,
		initMeetupMap: window.initMeetupMap
	};
})();
