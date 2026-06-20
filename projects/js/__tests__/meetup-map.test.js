// Polyfill TextEncoder and TextDecoder for Node.js environment
if (typeof global.TextEncoder === 'undefined') {
	const { TextEncoder, TextDecoder } = require('util');
	global.TextEncoder = TextEncoder;
	global.TextDecoder = TextDecoder;
}

// Mock Google Maps API
global.google = {
	maps: {
		Geocoder: jest.fn().mockImplementation(() => ({
			geocode: jest.fn((request, callback) => {
				if (request.location) {
					callback([
						{
							formatted_address: 'White City, London, UK',
							address_components: [
								{ long_name: 'White City', types: ['neighborhood'] },
							],
						},
					], 'OK');
					return;
				}
				const mockResult = {
					geometry: {
						location: {
							lat: () => 40.7128,
							lng: () => -74.0060,
						},
					},
				};
				callback([mockResult], 'OK');
			}),
		})),
		places: {
			Place: {
				searchByText: jest.fn(async (request) => ({
					places: [
						{
							displayName: 'Test Place 1',
							location: { lat: () => 40.7128, lng: () => -74.0060 },
							vicinity: '123 Test St',
							name: 'Test Place 1',
						},
					],
				})),
			},
			PlacesService: jest.fn().mockImplementation(() => ({
				nearbySearch: jest.fn((request, callback) => {
					const mockPlaces = [
						{
							name: 'Test Place 1',
							geometry: { location: { lat: () => 40.7128, lng: () => -74.0060 } },
							vicinity: '123 Test St',
						},
					];
					callback(mockPlaces, 'OK');
				}),
			})),
			Autocomplete: jest.fn(),
			PlaceAutocompleteElement: jest.fn().mockImplementation(() => {
				const node = document.createElement('div');
				node.value = '';
				node.addEventListener = jest.fn();
				node.focus = jest.fn();
				return node;
			}),
		},
		DirectionsService: jest.fn().mockImplementation(() => ({
			route: jest.fn((request, callback) => {
				const mockResponse = {
					routes: [
						{
							legs: [
								{
									distance: { value: 10000 },
									duration: { value: 600 },
								},
							],
						},
					],
				};
				callback(mockResponse, 'OK');
			}),
		})),
		DirectionsRenderer: jest.fn(),
		Map: jest.fn().mockImplementation(() => ({
			setCenter: jest.fn(),
			setZoom: jest.fn(),
			fitBounds: jest.fn(),
		})),
		LatLngBounds: jest.fn().mockImplementation(() => ({
			extend: jest.fn(),
		})),
		Circle: jest.fn().mockImplementation(() => ({
			setMap: jest.fn(),
		})),
		Marker: jest.fn(),
		InfoWindow: jest.fn(),
		marker: {
			AdvancedMarkerElement: jest.fn().mockImplementation(() => ({
				addEventListener: jest.fn(),
				set map(_) { },
			})),
		},
		TravelMode: {
			DRIVING: 'DRIVING',
			WALKING: 'WALKING',
			TRANSIT: 'TRANSIT',
			BICYCLING: 'BICYCLING',
		},
		routes: {
			Route: {
				computeRoutes: jest.fn(async () => ({
					routes: [
						{
							legs: [
								{
									distanceMeters: 10000,
									durationMillis: 600000,
								},
							],
							createPolylines: jest.fn(() => [
								{
									setMap: jest.fn(),
								},
							]),
						},
					],
				})),
			},
		},
	},
};
// Set up HTML fixture using Jest's built-in jsdom environment
document.body.innerHTML = `
  <div id="meetup-map"></div>
  <form id="meetup-form">
    <div id="location-list">
      <div class="location-row">
        <input name="location" type="text">
      </div>
    </div>
    <button type="button" id="add-location-button">Add another location</button>
    <select id="poi-category"></select>
    <select id="transport-mode"></select>
    <select id="optimization-strategy"></select>
    <button type="button" id="optimize-button">Optimize meetup</button>
    <button type="reset" id="reset-button">Reset</button>
  </form>
  <div id="results-container"></div>
`;

// Load the script
require('../meetup-map.js');

describe('Meetup Map Functions', () => {
	beforeEach(() => {
		jest.clearAllMocks();
		// Initialize the map and services by simulating initMeetupMap
		if (window.meetupMap && window.meetupMap.initMeetupMap) {
			try {
				window.meetupMap.initMeetupMap();
			} catch (e) {
				// Initialization may fail but we can still test individual functions
			}
		}
	});

	test('computeCentroid calculates average location', () => {
		const points = [
			{ lat: () => 40.0, lng: () => -74.0 },
			{ lat: () => 41.0, lng: () => -73.0 },
		];
		const centroid = window.meetupMap.computeCentroid(points);
		expect(centroid.lat).toBe(40.5);
		expect(centroid.lng).toBe(-73.5);
	});

	test('routeMetrics calculates distance and time', () => {
		const mockResponse = {
			routes: [
				{
					legs: [
						{
							distanceMeters: 5000,
							durationMillis: 300000,
						},
					],
				},
			],
		};
		const metrics = window.meetupMap.routeMetrics(mockResponse);
		expect(metrics.distanceMeters).toBe(5000);
		expect(metrics.durationSeconds).toBe(300);
		expect(metrics.distanceKm).toBe(5);
		expect(metrics.durationMinutes).toBe(5);
	});

	test('formatDuration formats seconds to readable string', () => {
		expect(window.meetupMap.formatDuration(300)).toBe('5 min');
		expect(window.meetupMap.formatDuration(3660)).toBe('1 hr 1 min');
	});

	test('formatDistance formats meters to readable string', () => {
		expect(window.meetupMap.formatDistance(500)).toBe('500 m');
		expect(window.meetupMap.formatDistance(1500)).toBe('1.5 km');
	});

	test('selectBestPlace selects place with lowest score', () => {
		const candidates = [
			{ place: { name: 'Place 1' }, scores: { distance: 1000 } },
			{ place: { name: 'Place 2' }, scores: { distance: 500 } },
		];
		const best = window.meetupMap.selectBestPlace(candidates, 'distance');
		expect(best.place.name).toBe('Place 2');
	});

	test('computeSeedCenter for two origins uses midpoint', () => {
		const origins = [
			{ lat: () => 40.0, lng: () => -74.0 },
			{ lat: () => 42.0, lng: () => -72.0 },
		];
		const seed = window.meetupMap.computeSeedCenter(origins);
		expect(seed.lat).toBe(41.0);
		expect(seed.lng).toBe(-73.0);
	});

	test('isPlaceInsideArea detects point inside radius', () => {
		const place = {
			geometry: {
				location: { lat: () => 40.7128, lng: () => -74.0060 },
			},
		};
		const area = {
			center: { lat: 40.7128, lng: -74.0060 },
			radiusMeters: 500,
		};
		expect(window.meetupMap.isPlaceInsideArea(place, area)).toBe(true);
	});

	test('selectTopPlaces returns lowest travel_time_first scores first', () => {
		const candidates = [
			{ place: { name: 'B' }, scores: { travel_time_first: 900 } },
			{ place: { name: 'A' }, scores: { travel_time_first: 300 } },
			{ place: { name: 'C' }, scores: { travel_time_first: 600 } },
		];
		const top = window.meetupMap.selectTopPlaces(candidates, 'travel_time_first', 2);
		expect(top).toHaveLength(2);
		expect(top[0].place.name).toBe('A');
		expect(top[1].place.name).toBe('C');
	});

	test('computeTravelTimeFirstScore penalizes imbalance and long max travel time', () => {
		const unfair = window.meetupMap.computeTravelTimeFirstScore({
			durationSeconds: 29 * 60,
			timeImbalanceSeconds: 27 * 60,
			maxDurationSeconds: 27 * 60,
		});
		const balanced = window.meetupMap.computeTravelTimeFirstScore({
			durationSeconds: 42 * 60,
			timeImbalanceSeconds: 5 * 60,
			maxDurationSeconds: 17 * 60,
		});
		expect(balanced).toBeLessThan(unfair);
	});

	test('formatDistance rounds meters', () => {
		expect(window.meetupMap.formatDistance(156.933)).toBe('157 m');
	});

	test('evaluateAreaCandidates falls back to estimated metrics when routing fails', async () => {
		global.google.maps.routes.Route.computeRoutes.mockRejectedValueOnce(new Error('route failed'));
		global.google.maps.routes.Route.computeRoutes.mockRejectedValueOnce(new Error('route failed'));
		const candidates = [{ lat: 40.7128, lng: -74.0060 }];
		const origins = [
			{ lat: () => 40.72, lng: () => -74.0 },
			{ lat: () => 40.70, lng: () => -74.01 },
		];
		const scored = await window.meetupMap.evaluateAreaCandidates(candidates, origins, 'driving');
		expect(scored).toHaveLength(1);
		expect(scored[0].score).toBeGreaterThan(0);
		expect(scored[0].perOrigin).toHaveLength(2);
		expect(scored[0].radiusMeters).toBeLessThanOrEqual(400);
	});

	test('reverseGeocodeAreaName returns neighborhood label when available', async () => {
		const areaName = await window.meetupMap.reverseGeocodeAreaName({ lat: 51.5, lng: -0.2 });
		expect(areaName).toBe('White City');
	});

	test('extractAreaNameFromGeocodeResults prefers specific labels for small radius', () => {
		const results = [
			{
				formatted_address: 'London, UK',
				address_components: [
					{ long_name: 'London', types: ['locality'] },
				],
			},
			{
				formatted_address: 'Charing Cross, London, UK',
				address_components: [
					{ long_name: 'Charing Cross', types: ['point_of_interest'] },
					{ long_name: 'London', types: ['locality'] },
				],
			},
		];
		const smallRadiusName = window.meetupMap.extractAreaNameFromGeocodeResults(results, 400);
		expect(smallRadiusName).toBe('Charing Cross');
	});

	test('extractAreaNameFromGeocodeResults prefers broader labels for larger radius', () => {
		const results = [
			{
				formatted_address: 'London, UK',
				address_components: [
					{ long_name: 'London', types: ['locality'] },
				],
			},
			{
				formatted_address: 'Charing Cross, London, UK',
				address_components: [
					{ long_name: 'Charing Cross', types: ['point_of_interest'] },
					{ long_name: 'London', types: ['locality'] },
				],
			},
		];
		const largeRadiusName = window.meetupMap.extractAreaNameFromGeocodeResults(results, 1200);
		expect(largeRadiusName).toBe('London');
	});

	test('geocodeAddress resolves with location', async () => {
		const location = await window.meetupMap.geocodeAddress('New York');
		expect(location.lat()).toBe(40.7128);
		expect(location.lng()).toBe(-74.0060);
	});

	test('loadPlaces resolves with places array', async () => {
		const places = await window.meetupMap.loadPlaces({ lat: 40.7128, lng: -74.0060 }, 'restaurant');
		expect(global.google.maps.places.Place.searchByText).toHaveBeenCalled();
		expect(places).toHaveLength(1);
		expect(places[0].name).toBe('Test Place 1');
	});

	test('getRoute resolves with directions response', async () => {
		const response = await window.meetupMap.getRoute(
			{ lat: 40.7128, lng: -74.0060 },
			{ lat: 40.7128, lng: -74.0060 },
			'DRIVING'
		);
		expect(response.routes).toBeDefined();
	});

	test('addLocationInput adds a new location row and attaches autocomplete', () => {
		const initialCount = document.querySelectorAll('.location-row').length;
		window.meetupMap.addLocationInput();
		expect(document.querySelectorAll('.location-row').length).toBe(initialCount + 1);
		expect(global.google.maps.places.PlaceAutocompleteElement).toHaveBeenCalled();
	});

	test('gm_authFailure displays an auth error message in the results container', () => {
		window.gm_authFailure();
		expect(document.getElementById('results-container').textContent).toMatch(
			'Google Maps authentication failed'
		);
	});
});