import dayjs, { Dayjs } from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

const WALL_CLOCK_FORMAT = "YYYY-MM-DDTHH:mm:ss";

export const ptuPickerToUtcIso = (value: Dayjs | null | undefined): string | null => {
  if (!value || typeof value.format !== "function") {
    return null;
  }
  return dayjs.utc(value.format(WALL_CLOCK_FORMAT)).toISOString();
};

export const utcIsoToPickerValue = (iso: string | null | undefined): Dayjs | null => {
  if (!iso) {
    return null;
  }
  return dayjs(dayjs.utc(iso).format(WALL_CLOCK_FORMAT));
};

const DISPLAY_FORMAT = "YYYY-MM-DD HH:mm:ss";

/**
 * Render a stored PTU timestamp for the read view. The backend serialises as `+00:00` while a
 * just-saved form holds the `.000Z` the picker produced, so the same instant would otherwise be
 * shown two different ways depending on whether the page has been reloaded since the edit. An
 * unparseable value is passed through rather than hidden.
 */
export const formatPtuUtcDisplay = (iso: string | null | undefined): string | null => {
  if (!iso) {
    return null;
  }
  const parsed = dayjs.utc(iso);
  return parsed.isValid() ? `${parsed.format(DISPLAY_FORMAT)} UTC` : String(iso);
};
