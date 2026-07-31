import dayjs from "dayjs";
import { describe, expect, it } from "vitest";
import { formatPtuUtcDisplay, ptuPickerToUtcIso, utcIsoToPickerValue } from "./ptuDatetime";

describe("ptuDatetime", () => {
  it("stores the picked wall-clock time as UTC instead of shifting across zones", () => {
    const picked = dayjs("2024-03-10T23:00:00");
    expect(ptuPickerToUtcIso(picked)).toBe("2024-03-10T23:00:00.000Z");
  });

  it("returns null for empty picker values", () => {
    expect(ptuPickerToUtcIso(null)).toBeNull();
    expect(ptuPickerToUtcIso(undefined)).toBeNull();
  });

  it("round-trips a UTC ISO string back to the same wall-clock in the picker", () => {
    const value = utcIsoToPickerValue("2024-03-10T23:00:00.000Z");
    expect(value).not.toBeNull();
    expect(value!.format("YYYY-MM-DDTHH:mm:ss")).toBe("2024-03-10T23:00:00");
    expect(ptuPickerToUtcIso(value)).toBe("2024-03-10T23:00:00.000Z");
  });

  it("returns null for empty ISO strings", () => {
    expect(utcIsoToPickerValue(null)).toBeNull();
    expect(utcIsoToPickerValue(undefined)).toBeNull();
    expect(utcIsoToPickerValue("")).toBeNull();
  });
});

describe("formatPtuUtcDisplay", () => {
  it("renders the two stored serialisations identically", () => {
    // the backend writes +00:00, a just-saved form holds the picker's .000Z
    expect(formatPtuUtcDisplay("2026-08-01T23:00:00+00:00")).toBe("2026-08-01 23:00:00 UTC");
    expect(formatPtuUtcDisplay("2026-08-01T23:00:00.000Z")).toBe("2026-08-01 23:00:00 UTC");
  });

  it("shows the UTC instant regardless of the offset it was written with", () => {
    expect(formatPtuUtcDisplay("2026-08-01T16:00:00-07:00")).toBe("2026-08-01 23:00:00 UTC");
  });

  it("returns null for empty values so the caller can fall back to Not Set", () => {
    expect(formatPtuUtcDisplay(null)).toBeNull();
    expect(formatPtuUtcDisplay(undefined)).toBeNull();
    expect(formatPtuUtcDisplay("")).toBeNull();
  });

  it("passes an unparseable value through rather than hiding it", () => {
    expect(formatPtuUtcDisplay("not-a-date")).toBe("not-a-date");
  });
});
