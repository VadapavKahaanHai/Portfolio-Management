from django.db import models

# Create your models here.

class Stock(models.Model):
    symbol     = models.CharField(max_length=20, unique=True)
    name       = models.CharField(max_length=100, null=True, blank=True)
    sector     = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.symbol

class StockPrice(models.Model):
    stock  = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="prices")
    date   = models.DateField()
    open   = models.FloatField(null=True)
    high   = models.FloatField(null=True)
    low    = models.FloatField(null=True)
    close  = models.FloatField(null=True)
    volume = models.BigIntegerField(null=True)

    class Meta:
        unique_together = ("stock", "date")
        ordering = ["-date"]

class StockReturn(models.Model):
    stock         = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="returns")
    date          = models.DateField()
    log_return    = models.FloatField(null=True)
    market_return = models.FloatField(null=True)

    class Meta:
        unique_together = ("stock", "date")
        ordering = ["-date"]

class MLPrediction(models.Model):
    stock            = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="predictions")
    predicted_on     = models.DateField(auto_now_add=True)
    predicted_return = models.FloatField(null=True)
    confidence       = models.FloatField(null=True)

    class Meta:
        ordering = ["-predicted_on"]

class PortfolioAllocation(models.Model):
    stock           = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="allocations")
    date            = models.DateField()
    weight          = models.FloatField()
    expected_return = models.FloatField(null=True)
    risk            = models.FloatField(null=True)

    class Meta:
        unique_together = ("stock", "date")
        ordering = ["-date"]