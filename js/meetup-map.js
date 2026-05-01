(function () {
	const maxLocations = 6;
	let map;
	let geocoder;
	let placeSearcher = null;
	let routeClass = null;
	let routePolylines = [];
	let markers = [];
	let centerMarker = null;
	let autocompletes = [];
	let previewDebounceTimer = null;
	const originColors = ['#d93025', '#1a73e8', '#188038', '#f9ab00', '#9334e6', '#00897b'];

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
		const autocomplete = createAutocomplete();
		newInput.parentNode.replaceChild(autocomplete, newInput);
		autocompletes.push(autocomplete);
		bindAutocompleteEvents(autocomplete);
		scheduleMapPreviewUpdate();
		try {
			autocomplete.focus();
		} catch (e) {
			// ignore
		}
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

	function loadPlaces(center, category) {
		return new Promise(async (resolve, reject) => {
			if (!placeSearcher?.searchByText) {
				reject(new Error('Place search is not available.'));
				return;
			}

			try {
				const request = {
					textQuery: category,
					locationBias: center,
					includedType: category,
					maxResultCount: 8,
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

			candidates.push({
				place,
				perOrigin,
				totals,
				scores: {
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
		if (meters >= 1000) {
			return `${(meters / 1000).toFixed(1)} km`;
		}
		return `${meters} m`;
	}

	function renderResults(candidate, strategy, origins) {
		const perOriginRows = candidate.perOrigin
			.map((metrics) => {
				const originLabel = origins[metrics.originIndex] || `Origin ${metrics.originIndex + 1}`;
				return `<tr>
				<td>${originLabel}</td>
				<td>${formatDistance(metrics.distanceMeters)}</td>
				<td>${formatDuration(metrics.durationSeconds)}</td>
				<td>${metrics.costDisplay ?? '–'}</td>
			</tr>`;
			})
			.join('');

		const html = `
      <h3>Best result: ${candidate.place.name}</h3>
      <p>${candidate.place.vicinity || candidate.place.formatted_address || ''}</p>
      <table>
        <thead>
          <tr>
            <th>Origin</th>
            <th>Distance</th>
            <th>Travel time</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          ${perOriginRows}
        </tbody>
      </table>
    `;

		$('#results-container').innerHTML = html;
	}

	function renderMap(center, origins, bestPlace) {
		if (!map) return;
		clearMarkers();
		map.setCenter(center);
		map.setZoom(11);

		origins.forEach((origin, index) => {
			const marker = createMarker(origin, `Origin ${index + 1}`, `${index + 1}`, 'location', getOriginColor(index));
			if (!marker) return;
			const infowindow = new google.maps.InfoWindow({ content: `Origin ${index + 1}` });
			marker.addEventListener('gmp-click', () => infowindow.open({ map, anchor: marker }));
		});

		if (bestPlace) {
			const highlighted = createMarker(bestPlace.place.geometry.location, bestPlace.place.name, '★', 'best-place');
			if (highlighted) {
				const infoContent = `<strong>${bestPlace.place.name}</strong><br>${bestPlace.place.vicinity || ''}`;
				const infowindow = new google.maps.InfoWindow({ content: infoContent });
				highlighted.addEventListener('gmp-click', () => infowindow.open({ map, anchor: highlighted }));
			}
		}
	}

	async function optimizeMeetup() {
		const origins = getOrigins();
		const category = $('#poi-category').value;
		const travelMode = $('#transport-mode').value;
		const strategy = $('#optimization-strategy').value;

		const results = $('#results-container');
		results.innerHTML = '<p>Computing the best meetup option... please wait.</p>';

		if (origins.length < 2) {
			results.innerHTML = '<p>Please enter at least two locations.</p>';
			return;
		}

		try {
			const geoPoints = await Promise.all(origins.map(geocodeAddress));
			const center = computeCentroid(geoPoints);
			const placeCandidates = await loadPlaces(center, category);
			const evaluatedPlaces = await evaluatePlaces(placeCandidates, geoPoints, travelMode);
			const best = selectBestPlace(evaluatedPlaces, strategy);
			if (!best) {
				throw new Error('Unable to choose a best place from the results.');
			}
			renderMap(center, geoPoints, best);
			await drawShortestRoutes(geoPoints, best.place.geometry.location, travelMode, strategy);
			renderResults(best, strategy, origins);
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
			const autocomplete = createAutocomplete();
			input.parentNode.replaceChild(autocomplete, input);
			autocompletes.push(autocomplete);
			bindAutocompleteEvents(autocomplete);
		});

		scheduleMapPreviewUpdate();

		attachEvents();
	};

	window.gm_authFailure = function () {
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
		loadPlaces,
		getRoute,
		routeMetrics,
		selectBestPlace,
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
