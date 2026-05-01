---
layout: default
title: Meetup Map
maps: true
description: Find the best halfway meeting point and points of interest for multiple locations.
---

## Meetup Map

Enter multiple starting locations, choose the category of interest, select a transport mode, and optimize by cost, distance, time, or recommended.

<div class="meetup-map-panel">
  <form id="meetup-form" class="meetup-form" aria-labelledby="meetup-form-title">
    <h2 id="meetup-form-title">Shared meetup planner</h2>

    <div id="location-list" class="location-list">
      <div class="location-row">
        <label for="location-1">Location 1</label>
        <input id="location-1" name="location" type="text" placeholder="Enter address or place" aria-label="Location 1" required>
      </div>
      <div class="location-row">
        <label for="location-2">Location 2</label>
        <input id="location-2" name="location" type="text" placeholder="Enter address or place" aria-label="Location 2" required>
      </div>
    </div>

    <button type="button" id="add-location-button" class="button-secondary">Add another location</button>

    <div class="field-row">
      <label for="poi-category">Place of interest</label>
      <select id="poi-category" aria-label="Category of place of interest">
        <option value="restaurant">Restaurant</option>
        <option value="cafe">Cafe</option>
        <option value="park">Park</option>
        <option value="museum">Museum</option>
        <option value="bar">Bar</option>
      </select>
    </div>

    <div class="field-row">
      <label for="transport-mode">Transport mode</label>
      <select id="transport-mode" aria-label="Mode of transport">
        <option value="driving">Driving</option>
        <option value="walking">Walking</option>
        <option value="transit">Transit</option>
        <option value="bicycling">Bicycling</option>
      </select>
    </div>

    <div class="field-row">
      <label for="optimization-strategy">Optimization strategy</label>
      <select id="optimization-strategy" aria-label="Optimization strategy">
        <option value="distance">Distance</option>
        <option value="time">Travel time</option>
        <option value="cost">Cost</option>
        <option value="recommended">Recommended</option>
      </select>
    </div>

    <div class="button-row">
      <button type="button" id="optimize-button">Optimize meetup</button>
      <button type="reset" id="reset-button" class="button-secondary">Reset form</button>
    </div>
  </form>

  <div class="meetup-side">
    <div id="meetup-map-canvas" class="meetup-map" role="application" aria-label="Meetup map showing midpoint and points of interest"></div>
    <div id="results-container" class="meetup-results" aria-live="polite"></div>
  </div>
</div>
