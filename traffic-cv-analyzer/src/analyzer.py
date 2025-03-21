def analyze_traffic(tracked_objects):
    stats = {
        "total_vehicles": len(tracked_objects),
        "active_tracks": [obj.id for obj in tracked_objects]
    }
    return stats
