import Time "mo:core/Time";
import Map "mo:core/Map";
import List "mo:core/List";
import Nat "mo:core/Nat";
import Text "mo:core/Text";
import Array "mo:core/Array";
import Order "mo:core/Order";
import Iter "mo:core/Iter";
import Runtime "mo:core/Runtime";
import Principal "mo:core/Principal";
import OutCall "http-outcalls/outcall";
import MixinAuthorization "authorization/MixinAuthorization";
import AccessControl "authorization/access-control";

actor {
  // Initialize the user system state
  let accessControlState = AccessControl.initState();
  include MixinAuthorization(accessControlState);

  public type BinId = Nat;
  public type Timestamp = Time.Time;
  public type Location = Text;

  public type BinStatus = {
    binId : BinId;
    fillLevel : Nat;
    location : Location;
    lastUpdated : Timestamp;
    capacity : Nat;
  };

  public type HistoricalData = {
    timestamp : Timestamp;
    fillLevel : Nat;
  };

  public type Route = {
    driverId : Principal;
    routeId : Nat;
    stops : [BinId];
    optimized : Bool;
    created : Timestamp;
  };

  public type Prediction = {
    binId : BinId;
    predictedFillLevel : Nat;
    timestamp : Time.Time;
  };

  module BinStatus {
    public func compare(bin1 : BinStatus, bin2 : BinStatus) : Order.Order {
      Nat.compare(bin1.binId, bin2.binId);
    };

    public func compareByFillLevel(bin1 : BinStatus, bin2 : BinStatus) : Order.Order {
      Nat.compare(bin2.fillLevel, bin1.fillLevel);
    };
  };

  let bins = Map.empty<BinId, BinStatus>();
  let historicalData = Map.empty<BinId, List.List<HistoricalData>>();
  let routes = Map.empty<Principal, List.List<Route>>();
  let driverAssignments = Map.empty<Principal, Principal>();
  var nextRouteId = 1;

  public type UserProfile = {
    name : Text;
    phone : Text;
    assignedRouteId : ?Nat;
  };

  let userProfiles = Map.empty<Principal, UserProfile>();

  ///////// Bin Management /////////

  public shared ({ caller }) func addBin(binId : BinId, capacity : Nat, location : Location) : async () {
    if (not (AccessControl.isAdmin(accessControlState, caller))) {
      Runtime.trap("Unauthorized: Only admins can add bins");
    };

    let bin : BinStatus = {
      binId;
      fillLevel = 0;
      capacity;
      location;
      lastUpdated = Time.now();
    };

    bins.add(binId, bin);
  };

  public shared ({ caller }) func updateBinStatus(binId : BinId, fillLevel : Nat) : async () {
    if (not (AccessControl.isAdmin(accessControlState, caller))) {
      Runtime.trap("Unauthorized: Only admins can update bin status");
    };

    switch (bins.get(binId)) {
      case (null) { Runtime.trap("Invalid bin id") };
      case (?bin) {
        let updatedBin : BinStatus = {
          binId;
          fillLevel;
          location = bin.location;
          capacity = bin.capacity;
          lastUpdated = Time.now();
        };

        bins.add(binId, updatedBin);

        // Add historical entry
        let entry : HistoricalData = {
          timestamp = Time.now();
          fillLevel;
        };

        let history = switch (historicalData.get(binId)) {
          case (null) { List.empty<HistoricalData>() };
          case (?existing) { existing };
        };

        // Add new entry to history
        history.add(entry);

        // Keep only last 100 entries
        let historyArray = history.toArray();
        let trimmed = if (history.size() > 100) {
          List.fromArray<[HistoricalData]>(historyArray.sliceToArray(0, 100));
        } else {
          history;
        };
        historicalData.add(binId, trimmed);
      };
    };
  };

  public query ({ caller }) func getAllBinStatuses() : async [BinStatus] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view bin statuses");
    };
    bins.values().toArray().sort(BinStatus.compareByFillLevel);
  };

  public query ({ caller }) func getBinStatus(binId : BinId) : async BinStatus {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view bin status");
    };
    switch (bins.get(binId)) {
      case (null) { Runtime.trap("Invalid bin Id. ") };
      case (?binStatus) { binStatus };
    };
  };

  public query ({ caller }) func getSortedBinsByFillLevel() : async [BinStatus] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view bin statuses");
    };
    bins.values().toArray().sort(BinStatus.compareByFillLevel);
  };

  public query ({ caller }) func getHistoricalData(binId : BinId) : async [HistoricalData] {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can view historical data");
    };
    switch (historicalData.get(binId)) {
      case (null) { [] };
      case (?history) { history.toArray() };
    };
  };

  ///////// Route Optimization /////////

  public shared ({ caller }) func createOptimizedRoute(driverId : Principal, binIds : [BinId]) : async Route {
    if (not (AccessControl.isAdmin(accessControlState, caller))) {
      Runtime.trap("Unauthorized: Only admins can create new routes for drivers");
    };

    let optimizedRoute : Route = {
      driverId;
      routeId = nextRouteId;
      stops = binIds;
      optimized = true;
      created = Time.now();
    };

    // Store route in driver's history
    let routeHistory = switch (routes.get(driverId)) {
      case (null) { List.empty<Route>() };
      case (?existing) { existing };
    };

    routeHistory.add(optimizedRoute);
    routes.add(driverId, routeHistory);

    nextRouteId += 1;
    optimizedRoute;
  };

  public query ({ caller }) func getDriverRoutes(driverId : Principal) : async [Route] {
    if (caller != driverId and not (AccessControl.isAdmin(accessControlState, caller))) {
      Runtime.trap("Unauthorized: Can only view your own routes");
    };
    switch (routes.get(driverId)) {
      case (null) { [] };
      case (?routeList) { routeList.toArray() };
    };
  };

  ///////// Driver Profile and Authentication /////////

  public query ({ caller }) func getCallerUserProfile() : async ?UserProfile {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can save profiles");
    };
    userProfiles.get(caller);
  };

  public query ({ caller }) func getUserProfile(user : Principal) : async ?UserProfile {
    if (caller != user and not (AccessControl.isAdmin(accessControlState, caller))) {
      Runtime.trap("Unauthorized: Can only view your own profile");
    };
    userProfiles.get(user);
  };

  public shared ({ caller }) func saveCallerUserProfile(profile : UserProfile) : async () {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can save profiles");
    };
    userProfiles.add(caller, profile);
  };

  ///////// Integrations /////////

  public query func transform(input : OutCall.TransformationInput) : async OutCall.TransformationOutput {
    OutCall.transform(input);
  };

  public shared ({ caller }) func fetchRouteFromGoogleMaps(origin : Location, destination : Location) : async Text {
    if (not (AccessControl.hasPermission(accessControlState, caller, #user))) {
      Runtime.trap("Unauthorized: Only users can fetch routes");
    };
    let url = "https://maps.googleapis.com/maps/api/directions/json?origin=" # origin # "&destination=" # destination # "&key=YOUR_API_KEY";
    await OutCall.httpGetRequest(url, [], transform);
  };
};
